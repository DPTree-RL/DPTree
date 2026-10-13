# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Optional, Union

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) # / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_tree_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   tree_index,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    treeid2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    treeid2mean = {}
    treeid2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            treeid2score[tree_index[i]].append(scores[i])
        for idx in treeid2score:
            if len(treeid2score[idx]) == 1:
                treeid2mean[idx] = torch.tensor(0.0)
                treeid2std[idx] = torch.tensor(1.0)
            elif len(treeid2score[idx]) > 1:
                treeid2mean[idx] = torch.mean(torch.tensor(treeid2score[idx]))
                treeid2std[idx] = torch.std(torch.tensor(treeid2score[idx]))
            else:
                raise ValueError(f"no score in prompt tree index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - treeid2mean[tree_index[i]]) / (treeid2std[tree_index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_grpo_per_token_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.clone()
    non_zero_mask = (token_level_rewards != 0)
    # scores = token_level_rewards * eos_mask

    id2tokens = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            valid_tokens = scores[i][eos_mask[i].bool()]
            id2tokens[index[i]].extend(valid_tokens.tolist())

        for idx in id2tokens:
            token_pool = torch.tensor(id2tokens[idx], device=scores.device)
            if token_pool.numel() == 1:
                id2mean[idx] = token_pool[0]
                id2std[idx] = torch.tensor(1.0)
            elif token_pool.numel() > 1:
                id2mean[idx] = torch.mean(token_pool)
                id2std[idx] = torch.std(token_pool)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores * eos_mask

    return scores, scores


def compute_tree_advantage(
    token_level_rewards: torch.Tensor,
    token_level_leaf_counts: torch.Tensor,
    token_level_step_first: torch.Tensor,
    response_mask: torch.Tensor,
    group_index,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Action-level normalized advantage for tree.

    Args:
        token_level_rewards:   [B, T]  raw r(a_k) = V(s_{k+1}) - V(s_k) per token
        token_level_leaf_counts: [B, T] int->float, subtree leaf count for each token's action
        token_level_step_first: [B, T] binary (long), 1 at the first token of each action range
        response_mask:         [B, T]  1 for valid response tokens
        group_index:           [B]     group id array (uid for inter-tree, tree_uid for inner-tree)
        epsilon:               numerical stability term

    Returns:
        advantages: [B, T]  (r - mu_group) / sigma_group / lc, masked by response_mask

    Statistics are computed at the **action level** (one r per unique action), with each shared
    non-leaf action importance-weighted by 1/lc so it counts exactly once regardless of how
    many trajectories pass through it.
    """
    B = token_level_rewards.shape[0]
    advantages = torch.zeros_like(token_level_rewards)

    alpha = 1  # TODO: hardcoded for now, may become configurable

    # --- Step 1: collect one (r, lc) per action per trajectory, grouped by group_index ---
    group2pairs = defaultdict(list)  # group_id -> list of (r, lc)

    with torch.no_grad():
        step_first_bool = token_level_step_first.bool()
        for i in range(B):
            gid = group_index[i]
            # positions that are valid response tokens AND the first token of an action step
            sel = step_first_bool[i] & response_mask[i].bool()
            if not sel.any():
                continue
            r_vals = token_level_rewards[i][sel]
            lc_vals = token_level_leaf_counts[i][sel]
            for r, lc in zip(r_vals.tolist(), lc_vals.tolist()):
                group2pairs[gid].append((r, lc))

    # --- Step 2: weighted mean and std per group (weight = 1/lc deduplicates shared actions) ---
    group2stats = {}
    for gid, pairs in group2pairs.items():
        r_t = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
        lc_t = torch.tensor([p[1] for p in pairs], dtype=torch.float32).clamp(min=1.0)
        lc_t = lc_t.pow(alpha)

        w = 1.0 / lc_t  # each unique node contributes total weight 1 across all trajectories

        total_w = w.sum()
        mu = (r_t * w).sum() / total_w
        # mu = torch.zeros_like(total_w) # RMS normalization
        var = (w * (r_t - mu) ** 2).sum() / (total_w - 1)
        sigma = torch.sqrt(var + epsilon)
        group2stats[gid] = (mu.item(), sigma.item())

    # --- Step 3: apply normalization token-wise ---
    with torch.no_grad():
        for i in range(B):
            gid = group_index[i]
            if gid not in group2stats:
                continue
            mu, sigma = group2stats[gid]
            valid = response_mask[i].bool()
            lc = token_level_leaf_counts[i].clamp(min=1.0).pow(alpha)
            advantages[i][valid] = (
                # (token_level_rewards[i][valid] - mu) / sigma / lc[valid]
                token_level_rewards[i][valid] / lc[valid]
            )

    return advantages


def compute_tree_per_token_advantage(
    token_level_rewards: torch.Tensor,
    token_level_leaf_counts: torch.Tensor,
    response_mask: torch.Tensor,
    group_index,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Token-level normalized advantage for tree_per_token.

    Like compute_tree_advantage but statistics are computed at the
    token level rather than the action level.  Shared (non-leaf) actions are
    deduplicated by assigning each token an importance weight of 1/lc, so a node
    shared by lc trajectories contributes the same total weight to the statistics
    as a unique node appearing in a single trajectory.

    Args:
        token_level_rewards:     [B, T]  raw r(a_k) per token
        token_level_leaf_counts: [B, T]  float, subtree leaf count for each token's action
        response_mask:           [B, T]  1 for valid response tokens
        group_index:             [B]     group id (uid for inter-tree, tree_uid for inner-tree)
        epsilon:                 numerical stability term

    Returns:
        advantages: [B, T]  (r - mu_group) / sigma_group / lc, masked by response_mask
    """
    B = token_level_rewards.shape[0]
    advantages = torch.zeros_like(token_level_rewards)

    # --- Step 1: collect weighted (r, w=1/lc) for all valid tokens per group ---
    group2r_w = defaultdict(list)  # group_id -> list of (r, w)

    with torch.no_grad():
        for i in range(B):
            gid = group_index[i]
            valid = response_mask[i].bool()
            if not valid.any():
                continue
            r_vals = token_level_rewards[i][valid]
            lc_vals = token_level_leaf_counts[i][valid].clamp(min=1.0)
            w_vals = 1.0 / lc_vals
            for r, w in zip(r_vals.tolist(), w_vals.tolist()):
                group2r_w[gid].append((r, w))

    # --- Step 2: weighted mean and std per group ---
    group2stats = {}
    for gid, r_w_list in group2r_w.items():
        r_t = torch.tensor([x[0] for x in r_w_list], dtype=torch.float32)
        w_t = torch.tensor([x[1] for x in r_w_list], dtype=torch.float32)

        total_w = w_t.sum()
        mu = (r_t * w_t).sum() / total_w
        var = (w_t * (r_t - mu) ** 2).sum() / total_w
        sigma = torch.sqrt(var + epsilon)
        group2stats[gid] = (mu.item(), sigma.item())

    # --- Step 3: apply per-token normalization ---
    with torch.no_grad():
        for i in range(B):
            gid = group_index[i]
            if gid not in group2stats:
                continue
            mu, sigma = group2stats[gid]
            valid = response_mask[i].bool()
            lc = token_level_leaf_counts[i]
            advantages[i][valid] = (
                (token_level_rewards[i][valid] - mu) / sigma / lc[valid]
            )

    return advantages


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss_turn(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    turns_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: Optional[Union[float, torch.Tensor]] = None,
    global_batch_size: Optional[Union[float, torch.Tensor]] = None,
    global_turn_count: Optional[Union[float, torch.Tensor]] = None,
):
    """
    Compute the clipped policy objective and related metrics for GSPO.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        dp_size, batch_num_tokens, global_batch_size, global_turn_count: forwarded to
            `agg_loss`; only used by the "-minibatch" modes.
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl_token = log_prob - old_log_prob
    negative_approx_kl_token = torch.clamp(negative_approx_kl_token, min=-20.0, max=20.0)

    ratio = torch.zeros_like(log_prob)
    negative_approx_kl_turn = torch.zeros_like(log_prob)
    for i in _present_turn_ids(turns_mask, response_mask.bool()):
        turn_mask_i = torch.logical_and((turns_mask == i), response_mask)
        negative_approx_kl_i = (negative_approx_kl_token * turn_mask_i).sum(dim=1)
        turn_token_count = turn_mask_i.sum(dim=1).clamp(min=1)
        negative_approx_kl_i = (negative_approx_kl_i / turn_token_count).unsqueeze(1)
        ratio_turn = torch.exp(negative_approx_kl_i)
        ratio = torch.where(turn_mask_i, ratio_turn, ratio)
        negative_approx_kl_turn = torch.where(turn_mask_i, negative_approx_kl_i, negative_approx_kl_turn)

    ppo_kl = (-negative_approx_kl_turn).mean()

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
                       dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size,
                       turns_mask=turns_mask, global_turn_count=global_turn_count)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_GSPO(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: Optional[Union[float, torch.Tensor]] = None,
    global_batch_size: Optional[Union[float, torch.Tensor]] = None,
):
    """
    Compute the clipped policy objective and related metrics for GSPO.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl_token = log_prob - old_log_prob
    negative_approx_kl_token = torch.clamp(negative_approx_kl_token, min=-20.0, max=20.0)

    seq_len = response_mask.sum(dim=1, keepdim=True).clamp(min=1)
    negative_approx_kl_seq = (negative_approx_kl_token * response_mask).sum(dim=1, keepdim=True) / seq_len

    ratio_seq = torch.exp(negative_approx_kl_seq)

    ppo_kl = (-negative_approx_kl_seq).mean()

    # GSPO-token (Zheng et al. 2025, "Group Sequence Policy Optimization", sec 4.5/eq 14):
    # s_i,t(theta) = sg[s_i(theta)] * (pi_theta(y_i,t) / sg[pi_theta(y_i,t)])
    # Forward VALUE: pi_theta(y_i,t)/sg[pi_theta(y_i,t)] == 1 exactly (a tensor over its own
    # detached copy), so s_i,t's value == sg[s_i(theta)] == ratio_seq (broadcast) -- clipping
    # below is decided at the sequence level, identical to plain GSPO.
    # GRADIENT: d/dtheta [pi_theta(y_i,t)/sg[pi_theta(y_i,t)]] = grad(log pi_theta(y_i,t)) at
    # this point (a plain exp(x - sg(x)) straight-through), so s_i,t's gradient only depends on
    # token t's own log_prob -- not the whole sequence's length-averaged log-ratio like plain
    # ratio_seq.expand_as(...) would. This lets per-token advantages (this project's tree reward
    # is token-level, not one scalar per sequence) get genuine per-token credit instead of having
    # their differences blurred together through the shared sequence-ratio gradient.
    # When every token in a sequence shares the same advantage (grpo/gspo's plain per-sequence
    # advantage), the aggregate gradient from this is provably identical to the old
    # ratio_seq.expand_as(...) -- so this is a strict generalization, not a behavior change for
    # that case.
    token_ratio_for_grad = torch.exp(log_prob - log_prob.detach())  # ==1 in value; carries this token's own grad
    ratio = ratio_seq.detach().expand_as(negative_approx_kl_token) * token_ratio_for_grad

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
                       dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_dual_clip(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: Optional[Union[float, torch.Tensor]] = None,
    global_batch_size: Optional[Union[float, torch.Tensor]] = None,
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
                       dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def _present_turn_ids(turns_mask: torch.Tensor, valid_mask: torch.Tensor) -> list:
    """Distinct turn ids actually present in turns_mask at valid (loss_mask-true) positions,
    as a plain Python list. Turn ids are positive integers assigned by the rollout loop (see
    generation_ts.py, one new id per step up to config.max_turns); 0 marks positions that
    aren't part of any turn. Derived dynamically -- one torch.unique() call, synced to host
    once -- rather than a hardcoded `range(1, 10)` upper bound, so this doesn't silently skip
    turns whenever max_turns is configured above whatever bound used to be hardcoded here (and
    is actually cheaper than the old loop, which forced a device sync on every one of its up to
    9 iterations via `.any()` inside the Python for-loop).
    """
    ids = torch.unique(turns_mask[valid_mask])
    return ids[ids > 0].tolist()


def _per_turn_means(loss_mat: torch.Tensor, loss_mask: torch.Tensor, turns_mask: torch.Tensor) -> torch.Tensor:
    """For each (sequence, turn) pair with at least one valid token, that pair's own token-mean
    of loss_mat over just that turn's valid tokens. Flat 1D, one entry per (sequence, turn)
    pair actually present -- the equal-weight units 'turn-mean'/'turn-mean-minibatch' average
    over, so a turn's contribution doesn't scale with how many tokens it happens to have (see
    compute_policy_loss_turn, which uses the same turn-id convention to build its per-turn
    ratio in the first place). loss_mask restricts to actually-valid response positions, same
    as every other agg_loss branch.
    """
    valid_mask = loss_mask.bool()
    per_turn = []
    for i in _present_turn_ids(turns_mask, valid_mask):
        turn_mask_i = torch.logical_and(turns_mask == i, valid_mask)
        has_turn = turn_mask_i.any(dim=-1)
        turn_token_count = turn_mask_i.sum(dim=-1).clamp(min=1)
        turn_loss = torch.sum(loss_mat * turn_mask_i, dim=-1) / turn_token_count
        per_turn.append(turn_loss[has_turn])
    if not per_turn:
        return loss_mat.new_zeros(0)
    return torch.cat(per_turn)


def count_turns(loss_mask: torch.Tensor, turns_mask: torch.Tensor) -> int:
    """Count (sequence, turn) pairs with at least one valid token -- the number of equal-weight
    units 'turn-mean-minibatch' needs summed across the whole mini-batch for its
    global_turn_count normalizer (see agg_loss). Mirrors the grouping in _per_turn_means
    without needing a loss_mat, since dp_actor.py only needs the count up front, before any
    micro-batch's loss has been computed.
    """
    valid_mask = loss_mask.bool()
    count = 0
    for i in _present_turn_ids(turns_mask, valid_mask):
        turn_mask_i = torch.logical_and(turns_mask == i, valid_mask)
        count += int(turn_mask_i.any(dim=-1).sum().item())
    return count


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str,
             dp_size: int = 1, batch_num_tokens: Optional[Union[float, torch.Tensor]] = None,
             global_batch_size: Optional[Union[float, torch.Tensor]] = None,
             turns_mask: Optional[torch.Tensor] = None,
             global_turn_count: Optional[Union[float, torch.Tensor]] = None):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
        dp_size: data parallel size. Only used by the "-minibatch" modes, to compensate for
            FSDP/DDP averaging gradients across dp ranks after backward (see those branches).
        batch_num_tokens: total valid token count for the whole mini-batch, summed across every
            micro-batch *and* every dp rank. Only used by "token-mean-minibatch"; required
            whenever dp_size > 1.
        global_batch_size: total count of sequences with at least one valid token, summed across
            every micro-batch *and* every dp rank. Only used by "seq-mean-token-mean-minibatch";
            required whenever dp_size > 1.
        turns_mask: `(torch.Tensor)`, shape (bs, response_length), integer turn id (1..9,
            matching compute_policy_loss_turn's convention) per token. Required by "turn-mean"
            and "turn-mean-minibatch".
        global_turn_count: total count of (sequence, turn) pairs with at least one valid token,
            summed across every micro-batch *and* every dp rank. Only used by
            "turn-mean-minibatch"; required whenever dp_size > 1.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss.
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "token-mean-minibatch":
        # True global-batch token-mean, ported from a newer verl's "token-mean": each
        # micro-batch's loss.backward() contributes masked_sum(this micro-batch) / batch_num_
        # tokens, and since batch_num_tokens is the *whole* mini-batch's token count (every
        # micro-batch, every dp rank), the accumulated gradient across all micro-batch backward
        # calls sums to (sum over the whole mini-batch) / batch_num_tokens -- a true token-mean,
        # not an average of per-micro-batch (or, without the dp_size factor here, per-dp-rank)
        # means, which is biased toward shorter micro-batches/ranks. The * dp_size compensates
        # for FSDP/DDP's automatic gradient *averaging* across dp ranks after backward, so the
        # reduced gradient ends up equal to true_sum / batch_num_tokens over every rank combined.
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        # +1e-8 and seq_valid guard against a fully-masked sequence (0 valid tokens) dividing by
        # zero and poisoning the whole-batch mean with NaN; see "seq-mean-token-mean-minibatch"
        # for the same fix applied to the global-batch variant.
        seq_token_counts = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_token_counts + 1e-8)  # token-mean
        seq_valid = (seq_token_counts > 0).float()
        loss = verl_F.masked_sum(seq_losses, seq_valid) / seq_valid.sum().clamp(min=1)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean-minibatch":
        # True global-batch seq-mean-token-mean, ported from a newer verl's "seq-mean-token-mean":
        # same per-sequence token-mean as "seq-mean-token-mean", but the outer seq-mean divides
        # by global_batch_size -- the count of valid sequences across the *whole* mini-batch
        # (every micro-batch, every dp rank) -- instead of just this call's own batch dimension,
        # so accumulated micro-batch backward() calls sum to a true equal-weight-per-sequence
        # average over the entire mini-batch (same reasoning as "token-mean-minibatch" above,
        # one level up). The +1e-8 and seq_valid mask also fix a latent bug in
        # "seq-mean-token-mean": a fully-masked sequence (0 valid tokens) there divides by zero
        # and poisons the whole-batch mean with NaN; here such sequences are excluded instead.
        seq_token_counts = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_token_counts + 1e-8)
        seq_valid = (seq_token_counts > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("(global) global_batch_size is required when dp_size > 1")
            global_batch_size = seq_valid.sum()
        loss = verl_F.masked_sum(seq_losses, seq_valid) / global_batch_size * dp_size
    elif loss_agg_mode == "turn-mean":
        # Give every turn equal weight regardless of its token count, unlike "token-mean" (which
        # -- since compute_policy_loss_turn assigns one ratio per turn, constant across that
        # turn's tokens -- ends up weighting turns by their token count) or "seq-mean-token-mean"
        # (which weights whole sequences equally but doesn't split a multi-turn sequence's own
        # turns apart). Local to this call only (like "token-mean"/"seq-mean-token-mean"); see
        # "turn-mean-minibatch" for the whole-mini-batch/dp-rank-global version.
        if turns_mask is None:
            raise ValueError("turns_mask is required for loss_agg_mode='turn-mean'")
        per_turn_losses = _per_turn_means(loss_mat, loss_mask, turns_mask)
        loss = per_turn_losses.mean() if per_turn_losses.numel() > 0 else loss_mat.sum() * 0.0
    elif loss_agg_mode == "turn-mean-minibatch":
        # Global (whole mini-batch, every micro-batch, every dp rank) version of "turn-mean",
        # same relationship as "token-mean" -> "token-mean-minibatch" above: divides by
        # global_turn_count instead of this call's own turn count, so accumulated micro-batch
        # backward() calls sum to a true equal-weight-per-turn average over the entire
        # mini-batch. The * dp_size compensates for FSDP/DDP's automatic gradient averaging
        # across dp ranks, same reasoning as the other "-minibatch" branches.
        if turns_mask is None:
            raise ValueError("turns_mask is required for loss_agg_mode='turn-mean-minibatch'")
        per_turn_losses = _per_turn_means(loss_mat, loss_mask, turns_mask)
        local_turn_loss_sum = per_turn_losses.sum() if per_turn_losses.numel() > 0 else loss_mat.sum() * 0.0
        if global_turn_count is None:
            if dp_size > 1:
                raise ValueError("(global) global_turn_count is required when dp_size > 1")
            global_turn_count = per_turn_losses.numel()
        loss = local_turn_loss_sum / global_turn_count * dp_size
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20.0, max=20.0)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
