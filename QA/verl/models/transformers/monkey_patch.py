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

#### Open Source Models
#### transformers version < 4.48


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


def apply_qwen3_5_gdn_patch(model_type: Optional[str], verbose: bool = True) -> bool:
    """Patch transformers' Qwen3.5 (dense) classes so cu_seqlens threads correctly through the
    Gated DeltaNet layers under this repo's rmpad/use_remove_padding path.

    Deliberately separate from apply_monkey_patch() above: that function is llama/qwen2-only,
    hard-asserts transformers is in [4.45.0, 4.47.1], and raises NotImplementedError for any
    other model_type -- calling it unconditionally for Qwen3.5 (which needs a much newer
    transformers) would break immediately. This one is self-contained and safe to call
    unconditionally for any model: it's a no-op for anything that isn't Qwen3.5, and the
    patched forwards behave identically to stock transformers when cu_seqlens is None (i.e.
    use_remove_padding off), so it doesn't need to be gated on that setting either.

    See verl/models/transformers/qwen3_5.py for the full rationale.
    """
    if model_type not in ('qwen3_5',):
        return False

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

    if verbose:
        print(f'Applying Qwen3.5 Gated DeltaNet cu_seqlens patch (model_type={model_type})')
    return True
