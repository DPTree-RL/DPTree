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

import functools
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
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad, slice_input_tensor
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

        # entropy_from_logits_with_chunking avoids materializing the full (tokens, vocab_size)
        # softmax at once (chunked along the token dim) -- ported from upstream verl, gated by
        # config so it can be toggled without touching this file again.
        entropy_from_logits_fn = (
            functools.partial(
                verl_F.entropy_from_logits_with_chunking,
                chunk_size=self.config.get('entropy_from_logits_chunk_size', 2048),
            )
            if self.config.get('entropy_from_logits_with_chunking', False)
            else verl_F.entropy_from_logits
        )
        self.compute_entropy_from_logits = torch.compile(entropy_from_logits_fn, dynamic=True)
        # entropy_checkpointing: recompute entropy on the backward pass instead of keeping its
        # activations, independent of (and stackable with) the model's own gradient checkpointing
        # -- also ported from upstream verl.
        self.entropy_checkpointing = self.config.get('entropy_checkpointing', False)

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
                    if self._pass_cu_seqlens:
                        # ulysses_pad_and_slice_inputs deliberately pads but does NOT slice
                        # position_ids_rmpad (its own docstring: "we don't need to slice position
                        # ids") -- correct for the old per-layer-rope patch style (qwen2/llama),
                        # where position_ids only ever reaches self.rotary_emb() while
                        # hidden_states is still this rank's own local shard, keeping both in
                        # sync. Qwen3.5 instead computes position_embeddings ONCE at the top of
                        # Qwen3_5TextModel.forward from (hidden_states, position_ids) and shares
                        # it across every layer -- there, an unsliced (full, padded) position_ids
                        # produces cos/sin sized to the *global* sequence while hidden_states/
                        # query_states are this rank's local shard, and apply_rotary_pos_emb
                        # blows up on the mismatch. Slice it here too, matching input_ids_rmpad's
                        # own local shard, so both stay in sync for Qwen3.5 specifically.
                        position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=-1, padding=False)
                        if pad_size > 0:
                            # cu_seqlens was computed by unpad_input() before any SP padding, so
                            # its last entry is the *unpadded* total_nnz. qwen3_5.py's GDN forward
                            # requires cu_seqlens[-1] to equal the padded total (so it divides
                            # evenly by sp_size and matches this rank's local shard * sp_size) --
                            # append the SP pad as one extra dummy "sequence" spanning the padded
                            # tail, mirroring upstream verl's transformer_impl.py.
                            padded_total = int(cu_seqlens[-1].item()) + pad_size
                            cu_seqlens = torch.cat((cu_seqlens, cu_seqlens.new_tensor([padded_total])))

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

                # compute entropy. Checkpointing only saves anything when a backward pass is
                # actually coming (update_policy) -- under compute_log_prob's torch.no_grad(),
                # there's nothing to recompute, so skip it there to avoid the pointless
                # "None of the inputs have requires_grad=True" checkpoint warning/overhead.
                entropy_rmpad = (
                    self.compute_entropy_from_logits(logits_rmpad)
                    if not (self.entropy_checkpointing and torch.is_grad_enabled())
                    else torch.utils.checkpoint.checkpoint(
                        self.compute_entropy_from_logits, logits_rmpad, use_reentrant=False
                    )
                )  # ((total_nnz / sp) + pad)

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
                entropy = (
                    verl_F.entropy_from_logits(logits)
                    if not (self.entropy_checkpointing and torch.is_grad_enabled())
                    else torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits, use_reentrant=False)
                )  # (bsz, response_length)

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
            #
            # Under ulysses SP, FSDPUlyssesShardingManager.preprocess_data() (see
            # fsdp_workers.py's update_actor) all-gathers each mini-batch across the SP group
            # before update_policy() ever runs, so every SP rank within a DP replica holds the
            # exact same (un-sharded) rows here -- the token-level sharding only happens later,
            # inside _forward_micro_batch's ulysses_pad_and_slice_inputs. That means the local
            # counts below are identical across all ulysses_sequence_parallel_size ranks of a
            # replica, so summing over the world process group (which contains those SP
            # duplicates) inflates the true global count by exactly that factor -- divide it back
            # out once after the all_reduce.
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
                if self.use_ulysses_sp:
                    batch_num_tokens /= self.ulysses_sequence_parallel_size
            elif is_seq_mean_minibatch:
                local_seq_count = 0.0
                for mb in micro_batches:
                    response_length = mb['responses'].size(1)
                    mb_response_mask = mb['loss_mask'] if self.config.state_masking else mb['attention_mask'][:, -response_length:]
                    local_seq_count += (mb_response_mask.sum(dim=-1) > 0).sum().item()
                global_batch_size = torch.tensor(local_seq_count, device='cuda')
                torch.distributed.all_reduce(global_batch_size, op=torch.distributed.ReduceOp.SUM)
                if self.use_ulysses_sp:
                    global_batch_size /= self.ulysses_sequence_parallel_size
            elif is_turn_mean_minibatch:
                local_turn_count = 0
                for mb in micro_batches:
                    response_length = mb['responses'].size(1)
                    mb_response_mask = mb['loss_mask'] if self.config.state_masking else mb['attention_mask'][:, -response_length:]
                    local_turn_count += core_algos.count_turns(mb_response_mask, mb['turns_mask'])
                global_turn_count = torch.tensor(float(local_turn_count), device='cuda')
                torch.distributed.all_reduce(global_turn_count, op=torch.distributed.ReduceOp.SUM)
                if self.use_ulysses_sp:
                    global_turn_count /= self.ulysses_sequence_parallel_size

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
                    # For the "-minibatch" modes, entropy_loss above is this micro-batch's own
                    # (diluted) contribution to the mini-batch-global gradient sum -- correct for
                    # backward(), but not a representative average on its own (see agg_loss's
                    # "-minibatch" branches: dividing by the *global* count means a mini-batch
                    # split into K micro-batches reports each one at ~1/K of the true value, and
                    # summing/averaging those across micro-batches for logging still lands at
                    # ~1/K, not the true average). Recompute the true LOCAL (this-micro-batch-
                    # only) average -- by omitting the global_* normalizer kwargs, agg_loss's
                    # "-minibatch" branches fall back to their own local count (see e.g.
                    # "token-mean-minibatch"'s `batch_num_tokens = loss_mask.sum()` when None),
                    # which is exactly the corresponding non-"-minibatch" mode's value -- purely
                    # for logging, so actor/entropy_loss is comparable across loss_agg_modes and
                    # doesn't drift with how many micro-batches a mini-batch happens to split
                    # into.
                    entropy_loss_for_metrics = core_algos.agg_loss(loss_mat=entropy, loss_mask=response_mask,
                                                                   turns_mask=turns_mask, loss_agg_mode=loss_agg_mode)
                else:
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)
                    entropy_loss_for_metrics = entropy_loss

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
                        # Metrics-only true local average -- see entropy_loss_for_metrics above
                        # for why the agg_loss call directly above isn't representative on its
                        # own for the "-minibatch" modes.
                        kl_loss_for_metrics = core_algos.agg_loss(loss_mat=kld, loss_mask=response_mask,
                                                                  turns_mask=turns_mask, loss_agg_mode=loss_agg_mode)
                    else:
                        kl_loss = masked_mean(kld, response_mask)
                        kl_loss_for_metrics = kl_loss

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss_for_metrics.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                # The "-minibatch" modes: pg_loss/entropy_loss/kl_loss above are already
                # normalized by the whole mini-batch's global count (agg_loss), so policy_loss
                # is already the correct per-micro-batch backward() contribution -- no further
                # division by the micro-batch count (that's implicitly handled by
                # batch_num_tokens/global_batch_size already covering every micro-batch, not
                # just this one).
                #
                # Non-"-minibatch" modes: each micro-batch's loss is a local (this-micro-batch-
                # only) mean, so it still needs dividing by how many micro-batches this mini-batch
                # was actually split into to average correctly across them. That must be
                # len(micro_batches) (the real count for *this* mini-batch), not the fixed
                # self.gradient_accumulation = ppo_mini_batch_size // ppo_micro_batch_size --
                # those only agree when use_dynamic_bsz is off (fixed-size splitting); with
                # dynamic_bsz on, rearrange_micro_batches packs by token budget, so the actual
                # count varies mini-batch to mini-batch and can differ from the configured one.
                loss = policy_loss if is_global_minibatch_mode else policy_loss / len(micro_batches)
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss_for_metrics.detach().item(),
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
