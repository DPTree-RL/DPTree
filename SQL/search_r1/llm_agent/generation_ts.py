import torch
import re
import json
import math
from collections import defaultdict, deque
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.utils.tree_search import compute_leaves_per_prompt
import shutil
import requests
import uuid
import random
import numpy as np

from .tree_node import TreeNode, DEBUG, dprint
from .sql_tools import (
    TOOL_NAMES,
    TOOL_REQUIRED_ARGS,
    READ_ONLY_METADATA_TOOLS,
    parse_tool_call,
)

# ---------------------------------------------------------------------------
# Terminal (leaf-only, original_score) outcomes -- set once per leaf, either when submit_sql is
# graded by the DB environment server, or when a rollout exhausts max_turns without ever
# submitting.
# ---------------------------------------------------------------------------
CORRECT_BASE_SCORE = 1.0
CORRECT_TURN_PENALTY = 0.0  # correct score = 1.0 - num_turns * 0.002
INCORRECT_SCORE = 0.0
INVALID_SUBMISSION_SCORE = -0.05
NO_SUBMISSION_SCORE = -0.10
# CORRECT_BASE_SCORE = 1.0
# CORRECT_TURN_PENALTY = 0.0  # correct score = 1.0 - num_turns * 0.002
# INCORRECT_SCORE = 0.0
# INVALID_SUBMISSION_SCORE = 0.0
# NO_SUBMISSION_SCORE = 0.0

# ---------------------------------------------------------------------------
# Process-level (additional_score, any turn) penalties -- set on every generated node based on
# how that single action (tool call) went. 0. for a clean/successful turn.
# ---------------------------------------------------------------------------
# MALFORMED_XML_PENALTY = -0.03
# INVALID_ARGUMENTS_PENALTY = -0.02
# UNSAFE_SQL_PENALTY = -0.05
# SQL_PARSE_ERROR_PENALTY = -0.03
# SQL_EXECUTION_ERROR_PENALTY = -0.02
# SQL_TIMEOUT_PENALTY = -0.03
MALFORMED_XML_PENALTY = 0.0
INVALID_ARGUMENTS_PENALTY = 0.0
UNSAFE_SQL_PENALTY = 0.0
SQL_PARSE_ERROR_PENALTY = 0.0
SQL_EXECUTION_ERROR_PENALTY = 0.0
SQL_TIMEOUT_PENALTY = 0.0

# Overlong-thinking penalty: once a single turn's <think>...</think> chain exceeds half of
# max_response_length, every token past that threshold costs THINK_LENGTH_PENALTY_CAP /
# (max_response_length / 2), capped at THINK_LENGTH_PENALTY_CAP once the chain reaches the full
# max_response_length. Threshold/per-token rate are derived from max_response_length per-run
# (see gen_action_chain), so only the cap is a fixed constant.
THINK_LENGTH_PENALTY_CAP = 0.4

_EXECUTE_ERROR_PENALTY = {
    'unsafe_sql': UNSAFE_SQL_PENALTY,
    'sql_parse_error': SQL_PARSE_ERROR_PENALTY,
    'sql_timeout': SQL_TIMEOUT_PENALTY,
    'sql_execution_error': SQL_EXECUTION_ERROR_PENALTY,
}

_ENDPOINT_FOR_TOOL = {
    'get_table_schema': '/table_schema',
    'get_column_meaning': '/column_meaning',
    'get_knowledge_definition': '/knowledge',
    'execute_sql': '/execute',
    'submit_sql': '/submit',
}

# Qwen3.5's chat template renders a `tool` role message (when the previous message was not
# itself a tool message) as:
#   <|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n
# and add_generation_prompt then appends '<|im_start|>assistant\n<think>\n'. Since our rollout
# never emits more than one tool call per turn, every observation is immediately preceded by an
# assistant turn -- so this fixed glue string, spliced onto the end of whatever the model just
# generated (which already ends right after </tool_call>, the vLLM stop string), reproduces
# exactly what apply_chat_template would have produced for this conversation shape.
_TOOL_RESPONSE_TEMPLATE = (
    "<|im_end|>\n<|im_start|>user\n<tool_response>\n{content}{note}\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
# A malformed_xml turn didn't produce a real tool call at all, so it's rendered as a plain user
# turn (no <tool_response> wrapper), matching Spider2's own corrective-nudge design.
_FORMAT_ERROR_MESSAGE_TEMPLATE = (
    "<|im_end|>\n<|im_start|>user\nYour previous message did not contain a valid tool call in "
    "the required format. Please respond again using exactly one "
    "<tool_call><function=...>...</function></tool_call> block.{note}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n"
)
# Matches the official LiveSQLBench-Agent harness's own turn-budget note (see
# ~/livesqlbench/LiveSQLBench-Agent/system_agent/callbacks.py's after_tool_callback:
# "\n\n[SYSTEM NOTE: Steps remaining: {steps_after}/{max_steps}]", appended to the tool response
# text after every interaction, but only while there's budget left -- no note once remaining
# hits 0, same rule kept here).
_TURNS_REMAINING_NOTE = "\n\n[SYSTEM NOTE: Turns remaining: {remaining}/{max_turns}]"


def _remaining_turns_note(remaining: int, max_turns: int) -> str:
    return _TURNS_REMAINING_NOTE.format(remaining=remaining, max_turns=max_turns) if remaining > 0 else ""


def _invalid_arguments_body(name: Optional[str], args: Dict[str, str]) -> str:
    if name not in TOOL_NAMES:
        return f"Unknown tool '{name}'. Available tools: {', '.join(sorted(TOOL_NAMES))}."
    missing = sorted(TOOL_REQUIRED_ARGS[name] - args.keys())
    return f"Missing required parameter(s) {missing} for tool '{name}'."


def _format_metadata_result(name: str, outcome: Dict[str, Any]) -> str:
    if name == 'get_table_schema':
        if outcome.get('schema') is None:
            return f"Not found: {outcome.get('error', 'no matching table')}"
        return outcome['schema']
    if name == 'get_column_meaning':
        meaning = outcome.get('meaning')
        if meaning is None:
            return f"Not found: {outcome.get('error', 'no matching column')}"
        # Most column meanings are plain strings, but JSONB columns store a nested dict
        # ({'column_meaning': ..., 'fields_meaning': {...}}) -- serialize non-str values like
        # every other branch here does, instead of handing a raw dict to the tokenizer.
        return meaning if isinstance(meaning, str) else json.dumps(meaning, ensure_ascii=False, indent=2)
    if name == 'get_knowledge_definition':
        if outcome.get('knowledge') is None:
            return f"Not found: {outcome.get('error', 'no matching knowledge entry')}"
        return json.dumps(outcome['knowledge'], ensure_ascii=False, indent=2)
    return json.dumps(outcome, ensure_ascii=False)


@dataclass
class GenerationTreeSearchConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    db_env_url: str = None
    m: int = 4  # number of trees
    n: int = 2  # number of nodes to expand
    l: int = 1  # number of iterations
    reward_mode: str = 'tree'
    expand_mode: str = 'random'

class LLMGenerationTreeSearchManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        reward_fn,
        config: GenerationTreeSearchConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        # reward_fn is unused for the SQL-agent task (leaf original_score is set inline during
        # rollout, at the exact turn submit_sql is graded by the DB environment server -- see
        # execute_predictions / gen_action_chain) but kept as a constructor arg for interface
        # compatibility with the trainer.
        self.reward_fn = reward_fn

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        self.root_list = []

        # '</think>' is a single, non-special token in Qwen3.5's vocab (survives
        # skip_special_tokens=True decode), used by _think_length_penalty to find where a turn's
        # chain-of-thought ends within its own generated token ids.
        self._think_end_token_id = self.tokenizer.convert_tokens_to_ids('</think>')

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Decode responses. vLLM already stops generation right after </tool_call> (see stop=
        in vllm_rollout_spmd_ts.py), so responses is already correctly truncated and does not
        need a decode -> truncate -> re-encode round trip."""
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""

        next_obs_ids = self.tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor,
                              next_obs_ids: torch.Tensor, step_prefix_ids: torch.Tensor = None) -> DataProto:
        """Update rolling state with new responses, observations, and step prefixes."""
        # Concatenate and handle padding
        if step_prefix_ids is not None:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids,
                step_prefix_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict(
            tensors={
                'input_ids': new_input_ids[:, -max_len:],
                'position_ids': new_position_ids[:, -max_len:],
                'attention_mask': new_attention_mask[:, -max_len:]
            }
        )
        new_rollings.non_tensor_batch = rollings.non_tensor_batch
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _info_masked_concatenate_with_padding(self,
                prompt: torch.Tensor,
                prompt_with_mask: torch.Tensor,
                response: torch.Tensor,
                turns_mask: torch.Tensor,
                turns: torch.Tensor,
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]

        # turns mask
        response_content_mask = (response != pad_id).to(dtype=turns_mask.dtype, device=turns_mask.device)
        turns_mask_tensors = [turns_mask, response_content_mask * turns.unsqueeze(1)]

        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
            turns_mask_tensors.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        concatenated_turns_mask = torch.cat(turns_mask_tensors, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        padded_turns_mask = concatenated_turns_mask.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info, padded_turns_mask

    def _update_right_side(self, right_side: Dict,
                           cur_responses: torch.Tensor,
                           turns: torch.Tensor,
                           next_obs_ids: torch.Tensor = None,
                           right_truncate: bool = True,
            ) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask, turns_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                right_side['turns_mask'],
                turns,
                next_obs_ids,
                pad_to_left=False
            )
        else:
            responses, responses_with_info_mask, turns_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                right_side['turns_mask'],
                turns,
                pad_to_left=False
            )

        per_row_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1)
        effective_len = per_row_len.max()
        max_len = min(self.config.max_prompt_length, effective_len)

        # Once a row's accumulated response+observation content exceeds max_prompt_length, this
        # truncation (right_truncate=True keeps the head, drops the tail) means every later turn's
        # content for that row gets discarded here before it ever reaches the training tensor --
        # not just this call's newly-added content, but permanently for the rest of the rollout,
        # since the accumulator never shrinks back under the cap. Silent otherwise.
        over_cap = per_row_len > self.config.max_prompt_length
        if over_cap.any():
            print(
                f"[WARNING] right_side accumulator exceeds max_prompt_length "
                f"({self.config.max_prompt_length} tokens) for {over_cap.sum().item()} row(s) -- "
                f"further turns for these rows will not be recorded in the training tensor."
            )

        if right_truncate:
            turns_mask = turns_mask[:, :max_len]
            responses = responses[:, :max_len]
            responses_with_info_mask = responses_with_info_mask[:, :max_len]
        else:
            turns_mask = turns_mask[:, -max_len:]
            responses = responses[:, -max_len:]
            responses_with_info_mask = responses_with_info_mask[:, -max_len:]

        return {'responses': responses, 'responses_with_info_mask': responses_with_info_mask, 'turns_mask': turns_mask}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def _get_action_log_probs(
            self,
            log_probs: torch.Tensor
        ):
        """
        Compute sum of log_probs for a batch of action steps.

        Args:
            log_probs: (batch_size, log_prob_length) log probabilities, padded with 0.0

        Returns:
            torch.Tensor: (batch_size,) sum of log_probs over valid tokens for each sample
        """
        return log_probs.sum(dim=-1)

    def _get_action_normalized_log_probs(
            self,
            log_probs: torch.Tensor
        ):
        """
        Compute mean log_prob over valid (non-padding) tokens for a batch of action steps.

        Args:
            log_probs: (batch_size, log_prob_length) log probabilities, padded with 0.0

        Returns:
            torch.Tensor: (batch_size,) mean log_prob over valid tokens for each sample
        """
        # Mask out padding (0.0) in log_probs
        # Valid tokens have non-zero log_probs (negative values in practice)
        log_probs_mask = (log_probs != 0.0).float()

        # Sum log_probs for valid tokens
        log_probs_sum = (log_probs * log_probs_mask).sum(dim=-1)

        # Count valid tokens per sample
        valid_token_counts = log_probs_mask.sum(dim=-1)

        # Compute mean, avoiding division by zero
        mean_log_probs = torch.where(
            valid_token_counts > 0,
            log_probs_sum / valid_token_counts,
            torch.zeros_like(log_probs_sum)
        )

        return mean_log_probs

    def _think_length_penalty(self, responses_ids: torch.Tensor) -> torch.Tensor:
        """Per-row overlong-thinking penalty for one turn's generated tokens.

        think_length = token index of this row's first '</think>' (i.e. token count of the
        chain-of-thought before it), or the row's full non-pad response length if '</think>'
        never appears (e.g. generation hit max_response_length before closing the think block --
        treat the whole thing as unclosed CoT).

        penalty = -min(THINK_LENGTH_PENALTY_CAP, max(0, think_length - threshold) * rate), where
        threshold = max_response_length / 2 and rate = THINK_LENGTH_PENALTY_CAP / threshold, so
        the penalty grows linearly from 0 at threshold to -THINK_LENGTH_PENALTY_CAP at
        max_response_length (and stays capped beyond that).

        Args:
            responses_ids: [n, T] this turn's generated token ids (pad_token_id-padded)
        Returns:
            [n] float tensor of penalties (<= 0)
        """
        is_think_end = responses_ids == self._think_end_token_id
        has_think_end = is_think_end.any(dim=1)
        # argmax on a bool/float mask returns the index of its first True (ties broken by lowest
        # index), which is exactly the first '</think>' position when one exists.
        think_end_pos = is_think_end.float().argmax(dim=1)
        response_lengths = (responses_ids != self.tokenizer.pad_token_id).sum(dim=1)
        think_lengths = torch.where(has_think_end, think_end_pos, response_lengths).float()

        threshold = self.config.max_response_length / 2
        rate = THINK_LENGTH_PENALTY_CAP / threshold
        excess = (think_lengths - threshold).clamp(min=0)
        return -(excess * rate).clamp(max=THINK_LENGTH_PENALTY_CAP)

    def gen_action_chain(
            self,
            gen_batch: DataProto,
            node_list: List[TreeNode],
            turns_stats: torch.Tensor,
        ):
        """
        Generate action chain for each example in gen_batch.

        Args:
            gen_batch (DataProto): gen_batch should include
                batch keys ['input_ids', 'attention_mask', 'position_ids']
                non_tensor_batch keys ['uid', 'data_source']. 'uid' also doubles as the DB
                environment task_id (see LLMGenerationTreeSearchManager._init_db_batch).
            node_list (list[TreeNode]): should have same order and same length as gen_batch
            turns_stats (torch.Tensor): turns_stats is the number of action turns for each example
        """
        task_ids = [str(u) for u in gen_batch.non_tensor_batch['uid']]

        # init original_right_side from node
        init_responses_list = []
        init_responses_with_info_mask_list = []
        init_turns_mask_list = []
        for i, node in enumerate(node_list):
            if node.responses is not None:
                responses = node.responses
                responses_with_info_mask = node.responses_with_info_mask
                turns_mask = node.turns_mask
            else:
                responses = torch.tensor([], dtype=gen_batch.batch['input_ids'].dtype).to(gen_batch.batch['input_ids'].device)
                responses_with_info_mask = torch.tensor([], dtype=gen_batch.batch['input_ids'].dtype).to(gen_batch.batch['input_ids'].device)
                turns_mask = torch.tensor([], dtype=gen_batch.batch['input_ids'].dtype).to(gen_batch.batch['input_ids'].device)
            init_responses_list.append(responses)
            init_responses_with_info_mask_list.append(responses_with_info_mask)
            init_turns_mask_list.append(turns_mask)

        init_responses = self.tensor_fn.pad_and_stack(init_responses_list, pad_to_left=False)
        init_responses_with_info_mask = self.tensor_fn.pad_and_stack(init_responses_with_info_mask_list, pad_to_left=False)
        init_turns_mask = self.tensor_fn.pad_and_stack(init_turns_mask_list, pad_to_left=False)
        original_right_side = {
            'responses': init_responses,
            'responses_with_info_mask': init_responses_with_info_mask,
            'turns_mask': init_turns_mask
        }

        tmp_node_list = node_list

        initial_input_ids = gen_batch.batch['input_ids'][:, -self.config.max_start_length:]

        active_mask = torch.ones(initial_input_ids.shape[0], dtype=torch.bool)
        valid_action_stats = torch.zeros(initial_input_ids.shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(initial_input_ids.shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        rollout_token_cnt = 0

        # fix bug. inherit parent node stats
        for i, node in enumerate(tmp_node_list):
            valid_action_stats[i] = node.valid_action_stats
            valid_search_stats[i] = node.valid_search_stats

        # Main generation loop
        for step in range(self.config.max_turns):
            # check if turns is reached
            # get need_act_mask by (turns_stats<config.max_turns)&active_mask
            need_act_mask = turns_stats < self.config.max_turns
            need_act_mask = need_act_mask.to(active_mask.dtype) * active_mask

            if not need_act_mask.sum():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict(
                    tensors={
                        k: v[need_act_mask] for k, v in rollings.batch.items()
                    },
                    meta_info=rollings.meta_info,
                )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            ## choice 1: get vllm log_probs for tree search, log_probs is a 2d tensor
            log_probs = gen_output.batch['infer_log_probs']
            log_probs = self.tensor_fn._example_level_pad_tensor(log_probs, need_act_mask)
            log_probs_step = self._get_action_log_probs(log_probs)
            normalized_log_probs_step = self._get_action_normalized_log_probs(log_probs)

            # meta_info = gen_output.meta_info
            # responses are padded with pad_token_id up to max_response_length regardless of how
            # many tokens were actually generated (see the DEBUG block below), so a row whose
            # non-pad count reaches the cap means vLLM's max_tokens cut it off mid-generation --
            # before it necessarily got to emit </tool_call> -- rather than it stopping on its own.
            non_pad_counts = (gen_output.batch['responses'] != self.tokenizer.pad_token_id).sum(dim=1)
            hit_cap = non_pad_counts >= self.config.max_response_length
            if hit_cap.any():
                active_task_ids = [tid for tid, m in zip(task_ids, need_act_mask.tolist()) if m]
                hit_task_ids = [tid for tid, h in zip(active_task_ids, hit_cap.tolist()) if h]
                print(
                    f"[WARNING] response hit max_response_length ({self.config.max_response_length} tokens) "
                    f"for {len(hit_task_ids)} row(s), task_ids={hit_task_ids[:10]}"
                    f"{'...' if len(hit_task_ids) > 10 else ''}"
                )

            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, need_act_mask)

            # for debug, stats response
            if DEBUG:
                # responses are padded with pad_token_id up to config.response_length regardless of how
                # many tokens were actually generated, so count non-pad tokens instead of shape[0]*shape[1]
                rollout_token_cnt += int((gen_output.batch['responses'] != self.tokenizer.pad_token_id).sum().item())

            # Execute the parsed tool call (if any) against the DB environment server and build
            # the next turn's observation. remaining_turns is turns left AFTER this one (turns_stats
            # itself only increments below, once we know which rows actually acted this turn).
            remaining_turns = (self.config.max_turns - (turns_stats + 1)).tolist()
            next_obs, dones, valid_action, is_execute_sql, additional_scores, submit_status = self.execute_predictions(
                responses_str, task_ids, need_act_mask, remaining_turns=remaining_turns
            )

            # Overlong-thinking penalty stacks additively on top of whatever execute_predictions
            # already scored this turn (malformed_xml / invalid_arguments / sql errors) -- it's an
            # independent process-level signal, not a replacement for those. Only meaningful for
            # rows that actually acted this turn.
            # think_penalties = self._think_length_penalty(responses_ids)
            # for i in range(len(additional_scores)):
            #     if need_act_mask[i]:
            #         additional_scores[i] += think_penalties[i].item()

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            curr_action_mask = need_act_mask
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[need_act_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_execute_sql, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)

            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                turns_stats,
                next_obs_ids,
            )

            # create tree node
            for i in range(gen_batch.batch['input_ids'].shape[0]):
                if not curr_action_mask[i]:
                    continue

                node_uid = str(uuid.uuid4())

                input_ids = rollings.batch['input_ids'][i]
                position_ids = rollings.batch['position_ids'][i]
                attention_mask = rollings.batch['attention_mask'][i]

                responses = original_right_side['responses'][i]
                responses_with_info_mask = original_right_side['responses_with_info_mask'][i]
                turns_mask = original_right_side['turns_mask'][i]

                # prompts not in rollings
                parent_node = tmp_node_list[i]
                prompts = parent_node.prompts
                tree_uid = parent_node.tree_uid

                new_node = TreeNode(
                    uid=parent_node.uid,
                    tree_uid=tree_uid,
                    node_uid=node_uid,
                    prompts=prompts,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    responses=responses,
                    responses_with_info_mask=responses_with_info_mask,
                    turns_mask=turns_mask,
                    log_prob=log_probs_step[i].item(),
                    normalized_log_prob=normalized_log_probs_step[i].item(),
                    parent_node=parent_node,
                    is_root=False,
                    is_active=bool(curr_active_mask[i].item()),
                    valid_action_stats=int(valid_action_stats[i].item()),
                    valid_search_stats=int(valid_search_stats[i].item()),
                    depth=parent_node.depth+1,
                    is_leaf=bool(dones[i]),
                    reward_mode=self.config.reward_mode,
                    tensor_fn=self.tensor_fn,
                )
                new_node.set_node_additional_score(additional_scores[i])
                status = submit_status[i]
                if status == 'correct':
                    new_node.set_leaf_original_score(CORRECT_BASE_SCORE - new_node.depth * CORRECT_TURN_PENALTY, status='correct')
                elif status == 'incorrect':
                    new_node.set_leaf_original_score(INCORRECT_SCORE, status='incorrect')
                elif status == 'invalid_submission':
                    new_node.set_leaf_original_score(INVALID_SUBMISSION_SCORE, status='invalid_submission')

                parent_node.add_child(new_node)
                tmp_node_list[i] = new_node

        # No forced extra "answer" turn: any row still active once the max_turns loop above ends
        # never called submit_sql -- its last generated node becomes a leaf with the
        # 'no_submission' terminal score.
        for i in range(gen_batch.batch['input_ids'].shape[0]):
            if active_mask[i]:
                node = tmp_node_list[i]
                node.is_leaf = True
                node.set_leaf_original_score(NO_SUBMISSION_SCORE, status='no_submission')

        dprint(f"ROLLOUT_TOKEN_CNT={rollout_token_cnt}")
        return active_num_list

    def run_llm_loop_tree_search(
            self,
            gen_batch,
            initial_input_ids: torch.Tensor,
        ) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        # gen_batch is one row per original prompt; all tree-search repetition (by m, n, and
        # the final leaf count) happens in here rather than the caller.
        original_gen_batch = gen_batch
        gen_batch = original_gen_batch.repeat(repeat_times=self.config.m, interleave=True)

        # All expand_mode's pick n new leaves per prompt per round out of one pool (see step 2
        # below): the m initial chains share a single root per prompt, so all of 'random'/
        # 'prob'/'atpo' pool candidates across that one whole tree.
        expand_gen_batch = original_gen_batch.repeat(repeat_times=self.config.n, interleave=True) if self.config.l > 0 else None

        # Every expansion attaches exactly one new leaf (candidates are always non-leaf nodes,
        # see TreeNode.get_candidate_nodes), so the final leaf count per prompt is deterministic
        # from m/n/l -- sample_all_leaves() takes every leaf, not a fixed k per tree.
        leaves_per_prompt = compute_leaves_per_prompt(self.config.m, self.config.n, self.config.l)
        final_output = original_gen_batch.repeat(repeat_times=leaves_per_prompt, interleave=True)

        # Provision one physical DB instance per distinct task (== uid) in this batch, shared by
        # every tree-search branch of that task and serialized server-side (see
        # search_r1/db_environment/server.py). Always clean up, even on error, so a failed step
        # doesn't leak DB instances into the next one.
        task_ids_per_prompt = self._init_db_batch(original_gen_batch)
        # One entry per gen_action_chain call (the initial m-chain call, then one per expansion
        # round): each entry is that call's own turn-by-turn active-trajectory count (its
        # active_mask starts fully active and only shrinks as branches call submit_sql).
        active_num_lists: List[List[int]] = []

        try:
            ## step 1. generate initial action chain (m,)
            turns_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
            # Create a single root per original prompt (not per repeated row): the m initial chains
            # are grown as m children of this one shared root, so the whole tree-search process
            # (expansion, normalization, reward) operates on one tree per prompt, not m separate
            # trees -- there is no longer an inter-tree vs intra-tree distinction.
            tmp_node_list = []
            for i in range(original_gen_batch.batch['input_ids'].shape[0]):
                uid = original_gen_batch.non_tensor_batch['uid'][i]
                tree_uid = str(uuid.uuid4())
                node_uid = str(uuid.uuid4())
                prompts = original_gen_batch.batch['prompts'][i]
                input_ids = original_gen_batch.batch['input_ids'][i]
                attention_mask = original_gen_batch.batch['attention_mask'][i]
                position_ids = original_gen_batch.batch['position_ids'][i]
                root_node = TreeNode(
                    uid=uid,
                    tree_uid=tree_uid,
                    node_uid=node_uid,
                    prompts=prompts,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    is_root=True,
                    is_active=True,
                    valid_action_stats=0,
                    valid_search_stats=0,
                    depth=0,
                    reward_mode=self.config.reward_mode,
                    tensor_fn=self.tensor_fn,
                )
                self.root_list.append(root_node)
                # gen_batch has this prompt repeated m times (interleaved); point all m of those
                # rows at the same root so gen_action_chain grows m children from it.
                tmp_node_list.extend([root_node] * self.config.m)
            # Then, get m initial chains from that one root per prompt
            active_num_lists.append(self.gen_action_chain(gen_batch, tmp_node_list, turns_stats))

            # step 2. Iteration to expand the trees
            for iteration in range(self.config.l):
                expansion_node_list = []

                # get expansion for n nodes. self.root_list has one root per prompt (see step 1
                # above), so no more inter-tree (across the m trees of a prompt) vs intra-tree
                # (within one tree) distinction -- every mode just pools this one tree's candidates.
                for root in self.root_list:
                    expansion_node_list_i = TreeNode.get_expand_node([root], self.config.n, mode=self.config.expand_mode)
                    expansion_node_list.extend(expansion_node_list_i)

                # prompts not need to pad
                prompts = torch.stack([node.prompts for node in expansion_node_list], dim=0)

                # need to pad
                input_ids = self.tensor_fn.pad_and_stack([node.input_ids for node in expansion_node_list])
                attention_masks = self.tensor_fn.create_attention_mask(input_ids)
                position_ids = self.tensor_fn.create_position_ids(attention_masks)

                turns_stats = torch.tensor([node.depth for node in expansion_node_list], dtype=torch.int)
                expand_gen_batch.batch['input_ids'] = input_ids
                expand_gen_batch.batch['attention_mask'] = attention_masks
                expand_gen_batch.batch['position_ids'] = position_ids

                # do the expansion
                active_num_lists.append(self.gen_action_chain(expand_gen_batch, expansion_node_list, turns_stats))

            # step 3. Get final output
            # The order is BS x M(trees) x K(samples per tree)
            # same as final_output
            final_responses_list = []
            final_responses_with_info_mask_list = []
            final_turns_mask_list = []
            final_tree_uid_list = []
            final_node_list = []
            for root in self.root_list:
                root.check_all_nodes_child()
                # BFS-order child count per node, one list per tree -- lets a debug run eyeball
                # each tree's branching shape (e.g. root's m initial children, then how many of
                # those got expanded and by how much) right before it gets flattened into leaves.
                # Training only (this same method also runs validation rollouts, via
                # self.is_validation) -- validation trees are pointless to inspect for branching
                # shape since expand_mode there is irrelevant to what's being measured.
                if not self.is_validation:
                    bfs_child_counts = []
                    queue = deque([root])
                    while queue:
                        node = queue.popleft()
                        bfs_child_counts.append(len(node.child_node))
                        queue.extend(node.child_node)
                    dprint(f"tree {root.tree_uid} bfs_child_counts={bfs_child_counts}")
                # sample trajectory
                final_node_list_tmp = root.sample_all_leaves()
                final_node_list.extend(final_node_list_tmp)

            for node in final_node_list:
                final_responses_list.append(node.responses)
                final_responses_with_info_mask_list.append(node.responses_with_info_mask)
                final_turns_mask_list.append(node.turns_mask)
                final_tree_uid_list.append(node.tree_uid)

            final_responses = self.tensor_fn.pad_and_stack(final_responses_list, pad_to_left=False)
            final_responses_with_info_mask = self.tensor_fn.pad_and_stack(final_responses_with_info_mask_list, pad_to_left=False)
            final_turns_mask = self.tensor_fn.pad_and_stack(final_turns_mask_list, pad_to_left=False)

            final_output.batch['responses'] = final_responses.long()
            final_output.batch['prompts'] = final_output.batch['prompts'].long()
            final_output.batch['input_ids'] = torch.cat([
                final_output.batch['prompts'],
                final_responses
            ], dim=1)
            final_output.batch['attention_mask'] = torch.cat([
                self.tensor_fn.create_attention_mask(final_output.batch['prompts']),
                self.tensor_fn.create_attention_mask(final_responses)
            ], dim=1)
            final_output.batch['info_mask'] = torch.cat([
                self.tensor_fn.create_attention_mask(final_output.batch['prompts']),
                self.tensor_fn.create_attention_mask(final_responses_with_info_mask)
            ], dim=1)
            final_output.batch['position_ids'] = self.tensor_fn.create_position_ids(
                final_output.batch['attention_mask']
            )
            final_output.batch['turns_mask'] = final_turns_mask

            # step 4. Compute scores and return outputs
            # Every leaf already has original_score set (from gen_action_chain, either via a
            # graded submit_sql or the no_submission fallback) -- no post-hoc reward_fn pass is
            # needed or used for this task.
            dprint('===========start calculate reward')
            final_output.batch['original_score'] = torch.tensor(
                [node.original_score for node in final_node_list], dtype=torch.float32
            )
            # normalize leaf scores and compute utility (+ normalized_additional_score for every
            # node) for each tree, grouped by uid (the original prompt group id) since
            # normalize_and_compute_utility standardizes leaf scores inter-tree, across all M
            # trees of the same prompt at once
            uid_to_roots = {}
            for root in self.root_list:
                uid_to_roots.setdefault(root.uid, []).append(root)
            # One 'all_correct'/'all_wrong'/'mixed'/'empty' label per prompt group -- surfaced as
            # env/group_all_correct, env/group_all_wrong, env/group_mixed counts (see
            # compute_data_metrics) to track how many groups this step actually had a
            # correct/wrong split to learn from vs. a uniform (zero-advantage-signal) outcome.
            group_type_list = []
            for group_roots in uid_to_roots.values():
                group_type_list.append(TreeNode.normalize_and_compute_utility(group_roots))

            # get final token-level score for each leaf
            if self.config.reward_mode == 'tree_softmax':
                # normalized_score is already standardized (z-score, inter-tree across the M
                # trees of this prompt) by normalize_and_compute_utility above, so leaf reward
                # starts there; dfs_compute_reward_softmax adds each node's own
                # normalized_additional_score and propagates a softmax(log_prob)-weighted
                # reward bottom-up through each tree.
                for node in final_node_list:
                    node.reward = node.normalized_score
                for root in self.root_list:
                    TreeNode.dfs_compute_reward_softmax(root)

            final_token_level_scores_list = []
            final_token_level_leaf_counts_list = []
            final_token_level_step_first_list = []
            for node in final_node_list:
                # 'tree_leafmean' reuses 'tree''s r(s) = V(s) + gamma*(V(s)-V(anchor)) + ...
                # formula verbatim -- only how V(s)/node.utility itself was computed differs
                # (see TreeNode.normalize_and_compute_utility / dfs_compute_utility_leafmean).
                if self.config.reward_mode in ('tree', 'tree_leafmean'):
                    scores, leaf_counts, step_first = node.get_token_level_score_tree()
                elif self.config.reward_mode == 'tree_softmax':
                    scores, leaf_counts, step_first = node.get_token_level_score_tree_softmax()
                else:
                    scores, leaf_counts, step_first = node.get_token_level_score_from_leaf_with_process()
                final_token_level_scores_list.append(scores)
                final_token_level_leaf_counts_list.append(leaf_counts)
                final_token_level_step_first_list.append(step_first)
            final_token_level_leaf_counts = self.tensor_fn.pad_and_stack(
                final_token_level_leaf_counts_list, pad_to_left=False, pad_value=1.0
            )
            final_token_level_step_first = self.tensor_fn.pad_and_stack(
                final_token_level_step_first_list, pad_to_left=False, pad_value=0.0
            )
            final_output.batch['token_level_leaf_counts'] = final_token_level_leaf_counts
            final_output.batch['token_level_step_first'] = final_token_level_step_first
            final_token_level_scores = self.tensor_fn.pad_and_stack(final_token_level_scores_list, pad_to_left=False, pad_value=0.0)
            final_output.batch['token_level_scores'] = final_token_level_scores

            # get meta info
            turns_stats = []
            active_mask = []
            valid_action_stats = []
            valid_search_stats = []
            for node in final_node_list:
                turns_stats.append(node.depth)
                active_mask.append(node.is_active)
                valid_action_stats.append(node.valid_action_stats)
                valid_search_stats.append(node.valid_search_stats)

            # number of children per node, across every node (root + all descendants) of every tree
            num_children_list = []
            for root in self.root_list:
                for node in [root] + root.get_subtree_nodes():
                    num_children_list.append(len(node.child_node))

            # probability (e^log_prob) of every action, across every non-root node (each
            # non-root node is one action) of every tree -- get_subtree_nodes() excludes the
            # root itself, which has no action/log_prob of its own.
            action_prob_list = []
            for root in self.root_list:
                for node in root.get_subtree_nodes():
                    action_prob_list.append(math.exp(node.log_prob))

            meta_info = {}
            meta_info['turns_stats'] = turns_stats
            meta_info['active_mask'] = active_mask
            meta_info['valid_action_stats'] = valid_action_stats
            meta_info['valid_search_stats'] = valid_search_stats
            meta_info['num_children_list'] = num_children_list
            meta_info['action_prob_list'] = action_prob_list
            meta_info['group_type_list'] = group_type_list
            meta_info['active_num_lists'] = active_num_lists

            final_output.meta_info = meta_info

            final_output.non_tensor_batch['tree_uid'] = np.array(final_tree_uid_list, dtype=object)
            # Raw 'correct'/'incorrect'/'invalid_submission'/'no_submission' label per leaf,
            # independent of whatever numeric original_score value those outcomes currently map
            # to -- see TreeNode.terminal_status. Consumed by ray_trainer_ts.py's _validate() for
            # the val/pass_rate, val/no_submission_rate, val/invalid_submission_rate and
            # val/avg_turns_success metrics.
            final_output.non_tensor_batch['terminal_status'] = np.array(
                [node.terminal_status for node in final_node_list], dtype=object
            )

            # Delete the tree
            for root in self.root_list:
                root.delete_tree_from_root()
            self.root_list = []

            return final_output
        finally:
            self._cleanup_db_batch(task_ids_per_prompt)

    # ------------------------------------------------------------------
    # DB environment server client: batch task provisioning/cleanup, and per-turn tool dispatch.
    # ------------------------------------------------------------------

    def _db_env_post(self, endpoint: str, payload: Dict, max_retries: int = 5, retry_delay: float = 2.0) -> Dict:
        url = self.config.db_env_url.rstrip('/') + endpoint
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        raise RuntimeError(
            f"DB environment request to {url} failed after {max_retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _to_jsonable(obj):
        """Recursively convert numpy types to native Python so json.dumps can serialize them.
        Needed because parquet/pyarrow round-trips nested list-valued dict fields (e.g.
        ground_truth.conditions.decimal) back as numpy.ndarray, not list."""
        if isinstance(obj, np.ndarray):
            return [LLMGenerationTreeSearchManager._to_jsonable(x) for x in obj.tolist()]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: LLMGenerationTreeSearchManager._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [LLMGenerationTreeSearchManager._to_jsonable(x) for x in obj]
        return obj

    def _init_db_batch(self, original_gen_batch: DataProto) -> List[str]:
        """POST /init_batch for every distinct task (== uid) in this training/validation step,
        so every tree-search branch of a task shares one physical DB instance for the whole
        step. Returns the per-prompt task_id list (== str(uid)), aligned to original_gen_batch's
        rows, so the caller can pass it to _cleanup_db_batch afterwards."""
        uids = [str(u) for u in original_gen_batch.non_tensor_batch['uid']]
        reward_models = original_gen_batch.non_tensor_batch['reward_model']
        tasks = []
        seen = set()
        for uid, rm in zip(uids, reward_models):
            if uid in seen:
                continue
            seen.add(uid)
            gt = rm.get('ground_truth', {}) if isinstance(rm, dict) else {}
            tasks.append({
                'task_id': uid,
                'selected_database': gt.get('selected_database'),
                'sol_sql': list(gt.get('sol_sql', [])),
                'test_cases': list(gt.get('test_cases', [])),
                'category': gt.get('category', 'Query'),
                'conditions': self._to_jsonable(dict(gt.get('conditions', {}))),
            })
        resp = self._db_env_post('/init_batch', {'tasks': tasks})
        errors = resp.get('errors', {})
        if errors:
            print(f"[WARNING] db_environment /init_batch errors: {errors}")
        return uids

    def _cleanup_db_batch(self, task_ids: List[str]):
        unique_ids = list(dict.fromkeys(task_ids))
        if not unique_ids:
            return
        try:
            self._db_env_post('/cleanup_batch', {'task_ids': unique_ids})
        except Exception as e:
            print(f"[WARNING] db_environment /cleanup_batch failed (instances may leak): {e}")

    def _dispatch_db_env_calls(self, jobs: List[Tuple[int, str, Dict[str, str]]], task_ids: List[str]) -> Dict[int, Dict]:
        """Send every well-formed tool call in this turn to the DB environment server
        concurrently (different tasks run fully in parallel server-side; same-task calls are
        serialized server-side via a per-task lock, see search_r1/db_environment/server.py)."""
        results: Dict[int, Dict] = {}
        if not jobs:
            return results

        def _one(job):
            i, name, args = job
            endpoint = _ENDPOINT_FOR_TOOL[name]
            payload = {'task_id': task_ids[i]}
            if name == 'get_table_schema':
                payload['table_name'] = args.get('table_name', '')
            elif name == 'get_column_meaning':
                payload['table_name'] = args.get('table_name', '')
                payload['column_name'] = args.get('column_name', '')
            elif name == 'get_knowledge_definition':
                payload['knowledge_name'] = args.get('knowledge_name')
            elif name in ('execute_sql', 'submit_sql'):
                payload['sql'] = args.get('sql', '')
            try:
                results[i] = self._db_env_post(endpoint, payload)
            except Exception as e:
                is_sql_tool = name in ('execute_sql', 'submit_sql')
                print(f"[WARNING] db_environment request failed for tool={name} task_id={task_ids[i]}: {e}")
                results[i] = {
                    'status': 'sql_execution_error' if is_sql_tool else 'error',
                    'success': False,
                    'error': f'DB environment request failed: {e}',
                }

        with ThreadPoolExecutor(max_workers=min(64, len(jobs))) as ex:
            list(ex.map(_one, jobs))
        return results

    def execute_predictions(
            self,
            predictions: List[str],
            task_ids: List[str],
            active_mask=None,
            remaining_turns: Optional[List[int]] = None,
        ) -> Tuple[List[str], List[int], List[int], List[int], List[float], List[Optional[str]]]:
        """
        Parse each active row's tool call and dispatch it to the DB environment server.

        Args:
            predictions: raw generated text per row (full-batch aligned)
            task_ids: DB environment task_id per row (== uid), full-batch aligned
            active_mask: bool/int array, full-batch aligned; inactive rows are no-ops
            remaining_turns: turns left AFTER this one, full-batch aligned (see gen_action_chain);
                appended to next_obs as a "[SYSTEM NOTE: Turns remaining: N/max]" note, matching
                the official LiveSQLBench-Agent harness's own budget note. None skips the note
                entirely (e.g. callers that don't track a turn budget).

        Returns:
            next_obs: text to splice onto the rolling context for the next turn
            dones: 1 iff this row just called submit_sql (episode ends, node becomes a leaf)
            valid_action: 1 iff a real, well-formed tool call was dispatched
            is_execute_sql: 1 iff the dispatched tool was execute_sql (diagnostic stat, mirrors
                the old QA task's "is_search" turn counter)
            additional_scores: process-level score for this turn's action (0. for a clean turn)
            submit_status: 'correct'/'incorrect'/'invalid_submission' iff this turn's action was
                a graded submit_sql, else None
        """
        n = len(predictions)
        next_obs = [''] * n
        dones = [0] * n
        valid_action = [0] * n
        is_execute_sql = [0] * n
        additional_scores = [0.0] * n
        submit_status: List[Optional[str]] = [None] * n

        dispatch_jobs: List[Tuple[int, str, Dict[str, str]]] = []
        for i in range(n):
            if active_mask is not None and not active_mask[i]:
                continue
            note = _remaining_turns_note(remaining_turns[i], self.config.max_turns) if remaining_turns is not None else ""
            name, args = parse_tool_call(predictions[i])
            if name is None:
                next_obs[i] = _FORMAT_ERROR_MESSAGE_TEMPLATE.format(note=note)
                additional_scores[i] = MALFORMED_XML_PENALTY
                continue
            if name not in TOOL_NAMES or not TOOL_REQUIRED_ARGS[name].issubset(args.keys()):
                next_obs[i] = _TOOL_RESPONSE_TEMPLATE.format(
                    content=f"Error: invalid arguments. {_invalid_arguments_body(name, args)}",
                    note=note,
                )
                additional_scores[i] = INVALID_ARGUMENTS_PENALTY
                continue
            dispatch_jobs.append((i, name, args))

        results = self._dispatch_db_env_calls(dispatch_jobs, task_ids)

        for i, name, args in dispatch_jobs:
            outcome = results.get(i, {})
            valid_action[i] = 1
            if name == 'submit_sql':
                dones[i] = 1
                status = outcome.get('status', 'invalid_submission')
                submit_status[i] = status
                message = outcome.get('message', outcome.get('error', ''))
                content = f"Submission result: {status}. {message}"
            elif name == 'execute_sql':
                is_execute_sql[i] = 1
                status = outcome.get('status')
                if status == 'ok':
                    content = outcome.get('result', '')
                else:
                    additional_scores[i] = _EXECUTE_ERROR_PENALTY.get(status, SQL_EXECUTION_ERROR_PENALTY)
                    content = f"Error: {outcome.get('error', 'unknown error')}"
            else:
                content = _format_metadata_result(name, outcome)

            content = self._truncate_text(content, self.config.max_obs_length, tool_name=name)
            note = _remaining_turns_note(remaining_turns[i], self.config.max_turns) if remaining_turns is not None else ""
            next_obs[i] = _TOOL_RESPONSE_TEMPLATE.format(content=content, note=note)

        return next_obs, dones, valid_action, is_execute_sql, additional_scores, submit_status

    def _truncate_text(self, text: str, max_tokens: int, tool_name: str = None) -> str:
        """Truncate text to at most max_tokens tokens (measured with self.tokenizer). This is
        the whole observation's budget for one action (get_table_schema result, execute_sql
        result table, error message, etc.) -- the DB environment server also caps its own
        responses, this is just a backstop so a single action's observation can never blow past
        max_obs_length regardless of what the server returns."""
        ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        if len(ids) <= max_tokens:
            return text
        print(f"[WARNING] observation truncated: tool={tool_name} {len(ids)} -> {max_tokens} tokens")
        return self.tokenizer.decode(ids[:max_tokens]) + "\n... [truncated]"
