# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
import torch.distributed
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

        # Qwen3.5's Gated DeltaNet layers need cu_seqlens explicitly passed through to stay
        # correct under rmpad (see verl/models/transformers/qwen3_5.py); other architectures'
        # forward() doesn't expect this kwarg at all, so only pass it when we know it's safe to.
        model_type = getattr(getattr(self.actor_module, 'config', None), 'model_type', None)
        self._pass_cu_seqlens = model_type in ('qwen3_5',)

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Qwen3.5's Gated DeltaNet layers need cu_seqlens to know sample boundaries
                # within this packed sequence, or the recurrent/conv state bleeds across
                # samples (see verl/models/transformers/qwen3_5.py). Other architectures'
                # forward() doesn't accept this kwarg, so only pass it for Qwen3.5.
                extra_kwargs = {'cu_seqlens': cu_seqlens} if self._pass_cu_seqlens else {}

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                    **extra_kwargs,
                )  # prevent model thinks we are generating

                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                )

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    entropy_rmpad = gather_outpus_and_unpad(
                        entropy_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    # **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                logits = output.logits

                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # self.actor_optimizer.step()
        ## if grad_norm is not finite, skip the updateAdd commentMore actions
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        is_token_mean_minibatch = self.config.loss_agg_mode == "token-mean-minibatch"
        is_seq_mean_minibatch = self.config.loss_agg_mode == "seq-mean-token-mean-minibatch"
        is_turn_mean = self.config.loss_agg_mode == "turn-mean"
        is_turn_mean_minibatch = self.config.loss_agg_mode == "turn-mean-minibatch"
        is_global_minibatch_mode = is_token_mean_minibatch or is_seq_mean_minibatch or is_turn_mean_minibatch
        # entropy_loss/kl_loss should stay on the same scale as pg_loss whenever pg_loss's own
        # aggregation isn't a plain per-token mean -- covers the "-minibatch" modes (global-
        # normalized) and plain "turn-mean" (per-turn-equal-weighted, but still local to this
        # call like "token-mean"/"seq-mean-token-mean" are).
        needs_agg_loss_for_entropy_kl = is_global_minibatch_mode or is_turn_mean
        if is_global_minibatch_mode and self.use_ulysses_sp:
            # batch_num_tokens/global_batch_size/global_turn_count below are summed via
            # all_reduce over the default (world) process group; with ulysses SP, that group
            # also contains SP ranks, which each hold only a slice of the same sequences (token
            # count) or see the exact same full, un-sharded sequences (sequence/turn count)
            # rather than independent samples, so the sum would double-count instead of
            # reflecting the true global batch. Not implemented.
            raise NotImplementedError(
                f"loss_agg_mode='{self.config.loss_agg_mode}' does not yet support "
                "ulysses_sequence_parallel_size > 1; use the non-minibatch variant instead.")
        dp_size = torch.distributed.get_world_size() if is_global_minibatch_mode else 1

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if self.config.policy_loss == 'turn':
            select_keys.append('turns_mask')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for data in dataloader:
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            # The "-minibatch" agg_loss modes need a global (every micro-batch, every dp rank)
            # normalizer computed once up front, before any micro-batch's loss is computed --
            # token-mean-minibatch needs the total valid token count, seq-mean-token-mean-
            # minibatch needs the count of sequences with at least one valid token, turn-mean-
            # minibatch needs the count of (sequence, turn) pairs with at least one valid token.
            # See core_algos.agg_loss for how each is used.
            batch_num_tokens = None
            global_batch_size = None
            global_turn_count = None
            if is_token_mean_minibatch:
                local_token_count = 0.0
                for mb in micro_batches:
                    response_length = mb['responses'].size(1)
                    mb_response_mask = mb['loss_mask'] if self.config.state_masking else mb['attention_mask'][:, -response_length:]
                    local_token_count += torch.sum(mb_response_mask).item()
                batch_num_tokens = torch.tensor(local_token_count, device='cuda')
                torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
            elif is_seq_mean_minibatch:
                local_seq_count = 0.0
                for mb in micro_batches:
                    response_length = mb['responses'].size(1)
                    mb_response_mask = mb['loss_mask'] if self.config.state_masking else mb['attention_mask'][:, -response_length:]
                    local_seq_count += (mb_response_mask.sum(dim=-1) > 0).sum().item()
                global_batch_size = torch.tensor(local_seq_count, device='cuda')
                torch.distributed.all_reduce(global_batch_size, op=torch.distributed.ReduceOp.SUM)
            elif is_turn_mean_minibatch:
                local_turn_count = 0
                for mb in micro_batches:
                    response_length = mb['responses'].size(1)
                    mb_response_mask = mb['loss_mask'] if self.config.state_masking else mb['attention_mask'][:, -response_length:]
                    local_turn_count += core_algos.count_turns(mb_response_mask, mb['turns_mask'])
                global_turn_count = torch.tensor(float(local_turn_count), device='cuda')
                torch.distributed.all_reduce(global_turn_count, op=torch.distributed.ReduceOp.SUM)

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = data['loss_mask']
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                turns_mask = data['turns_mask'] if self.config.policy_loss == 'turn' else None

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = (
                    self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                )
                clip_ratio_high = (
                    self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                )
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode
                # Shared by compute_policy_loss_GSPO/_turn/_dual_clip -- none of them need
                # turns_mask/global_turn_count for their own pg_loss (only compute_policy_loss_
                # turn's per-turn ratio does, and it takes turns_mask as its own dedicated
                # param below), so those two are passed separately where actually needed
                # (compute_policy_loss_turn, and the entropy/kl agg_loss calls) instead of
                # being bundled in here -- GSPO/dual_clip's signatures don't declare them.
                loss_agg_kwargs = dict(loss_agg_mode=loss_agg_mode, dp_size=dp_size,
                                       batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size)

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                if self.config.policy_loss == 'gspo':
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = core_algos.compute_policy_loss_GSPO(old_log_prob=old_log_prob,
                                                                                            log_prob=log_prob,
                                                                                            advantages=advantages,
                                                                                            response_mask=response_mask,
                                                                                            cliprange=clip_ratio,
                                                                                            cliprange_low=clip_ratio_low,
                                                                                            cliprange_high=clip_ratio_high,
                                                                                            clip_ratio_c=clip_ratio_c,
                                                                                            **loss_agg_kwargs,
                                                                                        )
                elif self.config.policy_loss == 'turn':
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = core_algos.compute_policy_loss_turn(old_log_prob=old_log_prob,
                                                                                            log_prob=log_prob,
                                                                                            advantages=advantages,
                                                                                            response_mask=response_mask,
                                                                                            turns_mask=turns_mask,
                                                                                            global_turn_count=global_turn_count,
                                                                                            cliprange=clip_ratio,
                                                                                            cliprange_low=clip_ratio_low,
                                                                                            cliprange_high=clip_ratio_high,
                                                                                            clip_ratio_c=clip_ratio_c,
                                                                                            **loss_agg_kwargs,
                                                                                        )
                else:
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = core_algos.compute_policy_loss_dual_clip(old_log_prob=old_log_prob,
                                                                                            log_prob=log_prob,
                                                                                            advantages=advantages,
                                                                                            response_mask=response_mask,
                                                                                            cliprange=clip_ratio,
                                                                                            cliprange_low=clip_ratio_low,
                                                                                            cliprange_high=clip_ratio_high,
                                                                                            clip_ratio_c=clip_ratio_c,
                                                                                            **loss_agg_kwargs,
                                                                                        )

                # compute entropy loss from entropy. The "-minibatch" modes and "turn-mean"
                # aggregate it the same way as pg_loss via agg_loss, instead of a plain
                # per-micro-batch mean, so the two stay on the same scale when combined below.
                if needs_agg_loss_for_entropy_kl:
                    entropy_loss = core_algos.agg_loss(loss_mat=entropy, loss_mask=response_mask,
                                                       turns_mask=turns_mask, global_turn_count=global_turn_count,
                                                       **loss_agg_kwargs)
                else:
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    if needs_agg_loss_for_entropy_kl:
                        kl_loss = core_algos.agg_loss(loss_mat=kld, loss_mask=response_mask,
                                                      turns_mask=turns_mask, global_turn_count=global_turn_count,
                                                      **loss_agg_kwargs)
                    else:
                        kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                # The "-minibatch" modes: pg_loss/entropy_loss/kl_loss above are already
                # normalized by the whole mini-batch's global count (agg_loss), so policy_loss
                # is already the correct per-micro-batch backward() contribution -- no further
                # division by gradient_accumulation (that's implicitly handled by
                # batch_num_tokens/global_batch_size already covering every micro-batch, not
                # just this one).
                loss = policy_loss if is_global_minibatch_mode else policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
