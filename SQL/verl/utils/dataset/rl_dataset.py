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

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 max_start_length=None,
                 max_turns=None,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        # The initial prompt (system tool-block + user query) is pad_token-padded to a fixed
        # width here so every row in a DataLoader batch has the same shape -- historically that
        # width was max_prompt_length (the same cap used for the whole rolling/right-side
        # trajectory later), which for this task is wildly oversized for a prompt that's
        # actually only ~1200-1400 tokens (measured across the full livesqlbench manifest): the
        # other ~28000+ columns are pure left-padding that rmpad strips before the model ever
        # sees it, but still inflates the raw tensor width that rearrange_micro_batches' pre-
        # check assertion (and any other code that looks at attention_mask.shape[-1] instead of
        # its real-content sum) measures. max_start_length is already required to be "large
        # enough the initial prompt is never truncated" (see generation_ts.py's use of it for
        # the rollout's starting context), so reusing it here as the padding target too shrinks
        # that waste without needing a new config knob. Falls back to max_prompt_length if not
        # given, matching the old behavior.
        self.max_start_length = max_start_length if max_start_length is not None else max_prompt_length
        # Data-processing scripts (e.g. scripts/data_process/livesqlbench_merge.py) can bake a
        # literal '__MAX_TURNS__' placeholder into the prompt text instead of a fixed number, so
        # the "you have at most N tool calls" hint shown to the agent always matches whatever
        # max_turns this run's launch script actually passes -- filled in below at render time
        # instead of at data-processing time, so it can't go stale.
        self.max_turns = max_turns
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)
        # pandas/pyarrow materializes list-of-dict parquet columns as numpy object arrays, not
        # native Python lists. Some chat templates (e.g. Qwen3.5's, which has many `is iterable`/
        # `is mapping` checks for multimodal content) choke on that with "truth value of an array
        # with more than one element is ambiguous" -- convert to plain list/dict up front.
        if hasattr(chat, 'tolist'):
            chat = chat.tolist()

        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False, enable_thinking=True
            )
            # Qwen3.5's template appends '<think>\n' itself whenever enable_thinking isn't
            # explicitly False. Qwen3 (non-.5)'s template has no such branch at all -- passing
            # enable_thinking=True there still emits nothing extra, since its template only ever
            # reacts to enable_thinking being explicitly False (a pre-closed empty think block);
            # it otherwise expects the model to open its own <think> tag unprompted. This project
            # treats a leading <think>\n as mandatory (see generation_ts.py's per-turn glue
            # strings, which always force one), so append it by hand whenever the template didn't
            # -- keeps turn 0 consistent with every later turn regardless of which Qwen family the
            # tokenizer belongs to.
            if not prompt_with_chat_template.endswith('<think>\n'):
                prompt_with_chat_template += '<think>\n'
            if self.max_turns is not None:
                prompt_with_chat_template = prompt_with_chat_template.replace(
                    '__MAX_TURNS__', str(self.max_turns)
                )
        else:
            prompt_with_chat_template = chat[0]['content']
        # prompt_with_chat_template = chat

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_start_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
