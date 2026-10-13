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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams, TokensPrompt
from verl.third_party.vllm import vllm_version

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRolloutTS(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        if torch.distributed.is_initialized():
            assert tensor_parallel_size <= torch.distributed.get_world_size(), \
                "tensor parallel size should be less than or equal to the world size"
        else:
            assert tensor_parallel_size == 1, \
                "In a non-distributed environment, tensor_parallel_size must be 1"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        # Composite/multimodal configs (e.g. Qwen3.5) nest the LLM's own fields under
        # .text_config; plain text-model configs already have them at the top level.
        text_hf_config = getattr(model_hf_config, 'text_config', None) or model_hf_config
        assert text_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            # Default "raw_logprobs" computes logprobs from the pre-temperature, pre-top_k/
            # top_p logits (see vllm/v1/sample/sampler.py's Sampler.forward), i.e. the model's
            # own raw distribution, not what was actually sampled from. "processed_logprobs"
            # instead reports logprobs from the post-temperature/top_k/top_p (renormalized)
            # distribution -- the one generation actually drew the token from. This only feeds
            # infer_log_probs -> TreeNode.log_prob/normalized_log_prob (used by get_expand_node's
            # 'prob'/'atpo' node-selection heuristics); PPO's old_log_prob is a separate,
            # unaffected recompute via actor_rollout_wg.compute_log_prob() (plain full-vocab
            # softmax, no top_p at all -- see ray_trainer_ts.py's fit()). Configurable per launch
            # script (actor_rollout_ref.rollout.logprobs_mode); defaults to vLLM's own default
            # ("raw_logprobs") when a script doesn't set it.
            logprobs_mode=self.config.get('logprobs_mode', 'raw_logprobs'),
            seed = 123
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            # stop generation right after </tool_call> (Qwen3.5's native XML tool-call format,
            # see search_r1/llm_agent/sql_tools.py) so the returned token_ids never need a
            # decode -> truncate -> re-encode round trip
            stop=['</tool_call>'],
            include_stop_str_in_output=True,
        )

        # Some model families (e.g. Qwen2/2.5) pad their embedding/lm_head to a
        # hardware-friendly vocab_size larger than the tokenizer's real vocab (e.g.
        # 151936 vs 151665 for Qwen2.5). Those extra ids have no tokenizer mapping;
        # if ever sampled they crash detokenization, or get fed back as next-turn
        # prompt tokens and crash vLLM's input validation ("Token id ... is out of
        # vocabulary"). vLLM V1 rejects per-request logit_bias/logits_processors
        # ("vLLM V1 does not support per request user provided logits processors"),
        # but allowed_token_ids is a native SamplingParams field with its own V1
        # support (not routed through the user-logits-processor path), so use it
        # as an explicit whitelist of every id the tokenizer actually knows.
        kwargs['allowed_token_ids'] = list(tokenizer.get_vocab().values())

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        prompts = prompts.to('cuda')
        prompts.batch = prompts.batch.cuda()
        

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                # LLM.generate() dropped the standalone prompt_token_ids kwarg -- token-id
                # prompts now go through prompts= wrapped in TokensPrompt.
                prompts=[TokensPrompt(prompt_token_ids=ids) for ids in idx_list],
                sampling_params=self.sampling_params,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        log_probs = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)
                
                log_probs_tmp = []
                for token_logprobs_dict in output.outputs[sample_id].logprobs:
                    numerical_values_per_token = list(token_logprobs_dict.values())[0].logprob
                    log_probs_tmp.append(numerical_values_per_token)
                log_probs.append(log_probs_tmp)

        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)
        log_probs = pad_2d_list_to_length(log_probs,
                                    0.0,
                                    max_length=self.config.response_length).to(idx.device)

        # if self.config.n > 1 and do_sample:
        #     idx = idx.repeat_interleave(self.config.n, dim=0)
        #     attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
        #     position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
        #     batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'infer_log_probs': log_probs, # for tree search
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
