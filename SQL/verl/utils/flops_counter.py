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

import torch
from transformers import PretrainedConfig, Qwen2Config, Qwen3Config, LlamaConfig

try:
    # Qwen3.5 requires transformers>=4.57 (a dev/pre-release version as of this writing) --
    # much newer than this repo's other Qwen2.5 pipelines are pinned to (4.51.3), so guard
    # the import instead of breaking everyone else's environment.
    from transformers import Qwen3_5Config, Qwen3_5TextConfig
    _QWEN3_5_CONFIG_TYPES = (Qwen3_5Config, Qwen3_5TextConfig)
except ImportError:
    _QWEN3_5_CONFIG_TYPES = ()

VALID_CONFIG_TYPE = (Qwen2Config, Qwen3Config, LlamaConfig) + _QWEN3_5_CONFIG_TYPES


def _count_qwen3_5_layer_types(text_config):
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        num_full_attn_layers = sum(layer_type == "full_attention" for layer_type in layer_types)
        num_linear_attn_layers = sum(layer_type == "linear_attention" for layer_type in layer_types)
        return num_full_attn_layers, num_linear_attn_layers

    full_attention_interval = getattr(text_config, "full_attention_interval", 4)
    num_full_attn_layers = sum(
        not bool((layer_idx + 1) % full_attention_interval) for layer_idx in range(text_config.num_hidden_layers)
    )
    return num_full_attn_layers, text_config.num_hidden_layers - num_full_attn_layers


def _compute_qwen3_5_hybrid_attn_params(text_config):
    hidden_size = text_config.hidden_size
    num_attention_heads = text_config.num_attention_heads
    num_key_value_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    num_full_attn_layers, num_linear_attn_layers = _count_qwen3_5_layer_types(text_config)

    # Qwen3.5 full attention q_proj also emits a sigmoid gate, so q_proj is 2x the query size.
    full_attn_linear_N = hidden_size * (2 * q_size + k_size + v_size + q_size)

    linear_k_size = text_config.linear_num_key_heads * text_config.linear_key_head_dim
    linear_v_size = text_config.linear_num_value_heads * text_config.linear_value_head_dim
    linear_attn_linear_N = hidden_size * (2 * linear_k_size + 3 * linear_v_size + 2 * text_config.linear_num_value_heads)
    conv_N = text_config.linear_conv_kernel_dim * (2 * linear_k_size + linear_v_size)

    attn_linear_N = full_attn_linear_N * num_full_attn_layers
    attn_linear_N += (linear_attn_linear_N + conv_N) * num_linear_attn_layers

    return attn_linear_N, num_full_attn_layers, num_linear_attn_layers, head_dim, num_attention_heads


def _compute_qwen3_5_gdn_recurrence_flops(text_config, tokens_sum, num_linear_attn_layers):
    return (
        15
        * text_config.linear_key_head_dim
        * text_config.linear_value_head_dim
        * text_config.linear_num_value_heads
        * tokens_sum
        * num_linear_attn_layers
    )


def get_device_flops(unit="T"):

    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device_name = torch.cuda.get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type
    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        if not isinstance(config, VALID_CONFIG_TYPE):
            print(f"Only support config type of {VALID_CONFIG_TYPE}, but got {type(config)}. "
                  f"MFU will always be zero.")

        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen3_5": self._estimate_qwen3_5_flops,
            "qwen3_5_text": self._estimate_qwen3_5_flops,
        }
        self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time):
        return 0

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        assert isinstance(self.config, (Qwen2Config, Qwen3Config, LlamaConfig))
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen3_5_flops(self, tokens_sum, batch_seqlens, delta_time):
        assert isinstance(self.config, _QWEN3_5_CONFIG_TYPES)
        # qwen3_5 (multimodal checkpoint) configs nest the LLM under .text_config; qwen3_5_text
        # (text-only checkpoint) configs are already the text config. This repo is text-only,
        # so unlike upstream verl there's no vision tower FLOPs term to add.
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        hidden_size = text_config.hidden_size
        vocab_size = text_config.vocab_size
        num_hidden_layers = text_config.num_hidden_layers

        attn_linear_N, num_full_attn_layers, num_linear_attn_layers, head_dim, num_attention_heads = (
            _compute_qwen3_5_hybrid_attn_params(text_config)
        )

        if hasattr(text_config, "num_experts"):
            moe_gate_N = hidden_size * text_config.num_experts
            moe_expertmlp_N = hidden_size * text_config.moe_intermediate_size * text_config.num_experts_per_tok * 3
            moe_sharedexpertmlp_N = hidden_size * text_config.shared_expert_intermediate_size * 3
            moe_sharedexpert_gate_N = hidden_size
            mlp_N = (moe_gate_N + moe_expertmlp_N + moe_sharedexpertmlp_N + moe_sharedexpert_gate_N) * num_hidden_layers
        else:
            mlp_N = hidden_size * text_config.intermediate_size * 3 * num_hidden_layers

        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N_flops = 6 * (mlp_N + attn_linear_N + emd_and_lm_head_N) * tokens_sum

        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_full_attn_layers

        gdn_recurrence_flops = _compute_qwen3_5_gdn_recurrence_flops(text_config, tokens_sum, num_linear_attn_layers)

        flops_all_token = dense_N_flops + attn_qkv_flops + gdn_recurrence_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
