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
Apply monkey-patch function to models
"""

from typing import Optional

import torch

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)

#### Open Source Models
#### transformers version < 4.48


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(batch, seqlen, num_key_value_heads, head_dim) -> (batch, seqlen, num_key_value_heads * n_rep, head_dim)."""
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention (DeepSpeed-Ulysses:
    https://arxiv.org/pdf/2309.14509). Ported from upstream verl's monkey_patch.py -- this is
    model-architecture-agnostic (it just intercepts the shared flash-attention entry point every
    modern transformers model routes through for attn_implementation="flash_attention_2"), so it
    covers Qwen3.5's periodic full-attention layers without any Qwen3.5-specific attention code.
    Qwen3.5's Gated DeltaNet (linear-attention) layers are handled separately, by the cp_context
    plumbing in qwen3_5.py -- this function is only reached for the full_attention layer type.

    position_ids is used as the signal for "is this a real (non-vision-encoder) attention call"
    -- this repo never runs the vision tower at all (see qwen3_5.py's _get_input_embeds), so that
    distinction doesn't actually matter here, but the check is kept for parity with upstream.
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1 and position_ids is not None:
        # Repeat kv heads to be divisible by sequence parallel size (flash attention supports
        # MQA/GQA, so we can repeat sp_size//nheads_k instead of the usual nheads_q//nheads_k).
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = _repeat_kv(key_states, repeats)
        value_states = _repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # (bsz, seq_len/n) -> (bsz, seq_len)
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

    query_length = query_states.size(1)
    from transformers.modeling_flash_attention_utils import _flash_attention_forward as _stock_flash_attention_forward
    attn_output = _stock_flash_attention_forward(
        query_states, key_states, value_states, attention_mask, query_length, *args, position_ids=position_ids, **kwargs
    )

    if ulysses_sp_size > 1 and position_ids is not None:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def apply_monkey_patch_to_llama():
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    from verl.models.transformers.llama import llama_flash_attn_forward
    LlamaFlashAttention2.forward = llama_flash_attn_forward


def apply_monkey_patch_to_qwen2():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2
    from verl.models.transformers.qwen2 import qwen2_flash_attn_forward
    Qwen2FlashAttention2.forward = qwen2_flash_attn_forward


_PATCH_NAME_TO_FUNC = {
    'llama': apply_monkey_patch_to_llama,
    'qwen2': apply_monkey_patch_to_qwen2,
}

from transformers import PretrainedConfig


def apply_monkey_patch(config: PretrainedConfig, verbose=True):
    if not is_transformers_version_in_range("4.45.0", "4.47.1"):
        raise AssertionError("The installed `transformers` version doesn't support ulysses patch. "
                             "Please install a version between 4.45.0 and 4.47.1 to use this ulysses feature.")
    success_apply_monkey_patch = False
    if config.model_type in _PATCH_NAME_TO_FUNC:
        _PATCH_NAME_TO_FUNC[config.model_type]()
        success_apply_monkey_patch = True

    if success_apply_monkey_patch and verbose:
        print(f'Applying monkey patch to model {config.model_type}')
    elif not success_apply_monkey_patch:
        raise NotImplementedError(f'Ulysses for model {config.model_type} is not implemented, \
                                   please set `ulysses_sequence_parallel_size=1`')

    return success_apply_monkey_patch


from functools import lru_cache
from packaging import version
import importlib.metadata


@lru_cache()
def is_transformers_version_in_range(min_version: str, max_version: str) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        raise ModuleNotFoundError("The `transformers` package is not installed.")

    # Check if the version is within the specified range
    return version.parse(min_version) <= version.parse(transformers_version) <= version.parse(max_version)


def apply_qwen3_5_gdn_patch(model_type: Optional[str], ulysses_sp_size: int = 1, verbose: bool = True, model_config=None) -> bool:
    """Patch transformers' Qwen3.5 (dense) classes so cu_seqlens threads correctly through the
    Gated DeltaNet layers under this repo's rmpad/use_remove_padding path.

    Also patches the shared flash-attention entry point every "full_attention"-type decoder
    layer's self_attn routes through (see _ulysses_flash_attention_forward above) with an
    Ulysses-aware version, so those layers stay correct under ulysses_sequence_parallel_size > 1
    too -- the Gated DeltaNet layers' own Ulysses support (the cp_context plumbing in
    qwen3_5_gated_delta_net_forward) doesn't cover them, since they're regular multi-head
    attention, not linear attention. Both patches are no-ops when SP isn't active.

    Deliberately separate from apply_monkey_patch() above: that function is llama/qwen2-only,
    hard-asserts transformers is in [4.45.0, 4.47.1], and raises NotImplementedError for any
    other model_type -- calling it unconditionally for Qwen3.5 (which needs a much newer
    transformers) would break immediately. This one is self-contained and safe to call
    unconditionally for any model: it's a no-op for anything that isn't Qwen3.5, and the
    patched forwards behave identically to stock transformers when cu_seqlens is None (i.e.
    use_remove_padding off) and ulysses_sequence_parallel_size == 1, so it doesn't need to be
    gated on either setting.

    See verl/models/transformers/qwen3_5.py for the full rationale.
    """
    if model_type not in ('qwen3_5',):
        return False

    if ulysses_sp_size > 1 and model_config is not None:
        text_config = getattr(model_config, 'text_config', None) or model_config
        num_attention_heads = text_config.num_attention_heads
        num_key_value_heads = text_config.num_key_value_heads
        assert num_attention_heads % ulysses_sp_size == 0, (
            f"num_attention_heads {num_attention_heads} must be divisible by "
            f"ulysses_sequence_parallel_size {ulysses_sp_size}"
        )
        assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (
            f"num_key_value_heads {num_key_value_heads} must be divisible by "
            f"ulysses_sequence_parallel_size {ulysses_sp_size} (or vice versa -- kv heads are "
            f"repeated to ensure correctness when ulysses_sp_size % num_key_value_heads == 0)"
        )

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5GatedDeltaNet,
        Qwen3_5Model,
    )
    from verl.models.transformers.qwen3_5 import (
        forward_with_normal_backend,
        qwen3_5_base_forward,
        qwen3_5_decoder_layer_forward,
        qwen3_5_gated_delta_net_forward,
    )

    Qwen3_5Model.forward = qwen3_5_base_forward
    Qwen3_5DecoderLayer.forward = qwen3_5_decoder_layer_forward
    Qwen3_5GatedDeltaNet.forward = qwen3_5_gated_delta_net_forward
    Qwen3_5ForConditionalGeneration.forward = forward_with_normal_backend

    # Qwen3_5DecoderLayer.self_attn (the "full_attention" layers, 1 in every
    # config.full_attention_interval) calls attn_implementation="flash_attention_2" the same way
    # every other modern transformers model does -- through ALL_ATTENTION_FUNCTIONS, which for
    # "flash_attention_2" resolves to transformers.integrations.flash_attention.
    # flash_attention_forward, which in turn calls the module-level name `_flash_attention_forward`
    # inside *that* module. Older transformers (<=4.47.1, or models still on the legacy pattern)
    # instead import `_flash_attention_forward` directly into the model's own module -- handle
    # both so this doesn't silently no-op if the installed transformers version differs.
    import transformers.models.qwen3_5.modeling_qwen3_5 as _qwen3_5_module
    if hasattr(_qwen3_5_module, '_flash_attention_forward'):
        _qwen3_5_module._flash_attention_forward = _ulysses_flash_attention_forward
        if verbose:
            print(f'Monkey patch _flash_attention_forward in {_qwen3_5_module.__name__}')
    else:
        from transformers.integrations import flash_attention
        flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
        if verbose:
            print(f'Monkey patch _flash_attention_forward in {flash_attention.__name__}')

    if verbose:
        print(f'Applying Qwen3.5 Gated DeltaNet cu_seqlens patch (model_type={model_type})')
    return True
