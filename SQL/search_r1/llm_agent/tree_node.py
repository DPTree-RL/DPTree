from __future__ import annotations

from typing import List, Optional, Callable
from pydantic import BaseModel
import json
import random
from collections import deque
import torch
from verl import DataProto
import random
import os

DEBUG: bool = os.environ.get('TREE_SEARCH_DEBUG', '').lower() == 'true'

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class TreeNode:
    def __init__(
        self,
        uid: str, # identifies the original prompt group; shared by all m trees of that prompt
        tree_uid: str, # identifies this single tree (unique per tree, not shared across the m trees of a prompt)
        node_uid: str,
        prompts: Optional[torch.Tensor] = None, # prompts are same as the original input_ids
        input_ids: Optional[torch.Tensor] = None, # input_ids are the left_part input during rollout
        attention_mask: Optional[torch.Tensor] = None, # attention_mask are the left_part input during rollout
        position_ids: Optional[torch.Tensor] = None, # position_ids are the left_part input during rollout
        responses: Optional[torch.Tensor] = None, # responses are the right_part output during rollout
        responses_with_info_mask: Optional[torch.Tensor] = None, # responses_with_info_mask are the right_part output during rollout
        turns_mask: Optional[torch.Tensor] = None, # turn_mask are the right_part output during rollout
        log_prob: Optional[float] = None,
        normalized_log_prob: Optional[float] = None,
        parent_node: Optional['TreeNode'] = None,
        is_root: bool = False,
        is_active: bool = True,
        valid_action_stats: int = 0,
        valid_search_stats: int = 0,
        depth: int = 0,
        is_leaf: bool = False,
        correct_leaf_in_subtree: int = 0,
        reward_mode: str = 'tree',
        tensor_fn = None,
    ):

        self.uid: str = uid
        self.tree_uid: int = tree_uid
        self.node_uid: int = node_uid

        self.prompts: torch.Tensor = prompts
        self.input_ids: torch.Tensor = input_ids
        self.attention_mask: torch.Tensor = attention_mask
        self.position_ids: torch.Tensor = position_ids
        self.responses: torch.Tensor = responses
        self.responses_with_info_mask: torch.Tensor = responses_with_info_mask
        self.turns_mask: torch.Tensor = turns_mask

        self.log_prob: Optional[float] = log_prob
        self.normalized_log_prob: Optional[float] = normalized_log_prob

        self.parent_node = parent_node
        self._child_node = []

        self.is_root = is_root
        self.is_active = is_active
        self.valid_action_stats = valid_action_stats
        self.valid_search_stats = valid_search_stats
        self.depth = depth
        self.is_leaf = is_leaf

        # Terminal (leaf-only) score: set once, when a leaf is finalized (submit_sql graded, or
        # the episode ends without a submission). See set_leaf_original_score. terminal_status is
        # the raw 'correct'/'incorrect'/'invalid_submission'/'no_submission' label this score came
        # from -- kept alongside the score itself so callers that need to tell these outcomes
        # apart (e.g. validation's no_submission_rate/invalid_submission_rate metrics) don't have
        # to infer it by comparing original_score against the *current* score constants, which
        # breaks silently if those constants are ever changed to collide (e.g. no_submission_score
        # and invalid_submission_score both set to 0.).
        self.original_score = 0.
        self.terminal_status: Optional[str] = None
        self.normalized_score = 0.

        # Process-level score: may be set on ANY node (leaf or not), reflecting the outcome of
        # the single action that produced this node from its parent (malformed_xml,
        # invalid_arguments, unsafe_sql, sql_parse_error, sql_execution_error, sql_timeout, or
        # 0. for a clean turn). See set_node_additional_score.
        self.additional_score = 0.
        self.normalized_additional_score = 0.

        self.utility = 0.
        self.reward = 0.
        self.subtree_leaf_num = 0

        self.reward_mode = reward_mode

        self.tensor_fn = tensor_fn

    @property
    def child_node(self) -> list['TreeNode']:
        return self._child_node

    @child_node.setter
    def child_node(self, value: list['TreeNode']):
        """
        Debug
        """
        print(f"!!! WARNING: Direct assignment to child_node on node {self.node_uid}. New list has {len(value)} children.")

        import traceback
        traceback.print_stack()

        # Check
        for child in value:
            if child is self:
                raise ValueError(f"CRITICAL: Attempted to directly assign a list containing the node itself as its own child. Node UID: {self.node_uid}")

        self._child_node = value

    def add_child(self, child_node: 'TreeNode'):
        if child_node is self:
            raise ValueError("A node cannot be its own child")
        if child_node.node_uid == self.node_uid:
            raise ValueError("node_uid same!! A node cannot be its own child")
        self._child_node.append(child_node)

    def get_subtree_nodes(self):
        """
        Dynamically get all descendant nodes by traversing the tree
        """
        nodes = []
        nodes_to_visit = list(self.child_node) # from child nodes
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            nodes.append(current_node)
            nodes_to_visit.extend(current_node.child_node)
        return nodes

    def get_candidate_nodes(self) -> List['TreeNode']:
        """
        Non-leaf nodes in this tree (including self) eligible for expansion
        """
        candidate_set = [self]
        for node in self.get_subtree_nodes():
            if not node.is_leaf:
                candidate_set.append(node)
        return candidate_set

    @staticmethod
    def _logsumexp(vals: List[float]) -> float:
        """log(sum(exp(v) for v in vals)). Empty sum is -inf."""
        if not vals:
            return float('-inf')
        return torch.logsumexp(torch.tensor(vals), dim=0).item()

    @staticmethod
    def _logsumsqrtexp(vals: List[float]) -> float:
        """log(sum(sqrt(exp(v)) for v in vals)) = logsumexp([v/2 for v in vals]). Empty sum is
        -inf. Alternative to _logsumexp: taking sqrt(prob) before summing compresses the
        dynamic range across vals, so one high-probability value dominates the sum less."""
        return TreeNode._logsumexp([v / 2 for v in vals])

    @staticmethod
    def _log_sum_child_probs(node: 'TreeNode') -> float:
        """log(sum(prob(child) for child in node.child_node)), used by 'prob'.
        Empty sum (no children explored yet) is -inf, so an unexplored node always sorts first."""
        return TreeNode._logsumexp([child.log_prob for child in node.child_node])

    @staticmethod
    def _min_child_log_prob(node: 'TreeNode') -> float:
        """min(normalized_log_prob(child) for child in node.child_node), used by 'atpo'.
        Length-normalized (mean per-token log_prob) rather than raw summed log_prob, so
        longer actions aren't penalized just for having more tokens.
        No children explored yet -> 0. (normalized_log_prob is always <= 0, so this matches
        the old incrementally-maintained sentinel default that any real update would beat)."""
        if not node.child_node:
            return 0.
        return min(child.normalized_log_prob for child in node.child_node)

    @staticmethod
    def _branch_anchor(node: 'TreeNode') -> 'TreeNode':
        """Anchor for get_token_level_score_tree's r(s) = V(s) + gamma*(V(s) - V(anchor)) term:
        walk up from node's parent to the nearest ancestor with more than one child (an actual
        branch point in the tree), or the root if no such ancestor exists. Most nodes along an
        unbranched straight chain have exactly 1 child, so using the immediate parent as anchor
        (the old behavior) compared a node's utility against another node from the same
        chain/branch -- not a meaningful "what if this action hadn't been taken" baseline, and
        let per-step utility noise compound into large reward swings step by step down a long
        chain. Anchoring at the nearest real decision point instead means every node between two
        branch points shares the same, more stable comparison baseline."""
        ancestor = node.parent_node
        while len(ancestor.child_node) <= 1 and not ancestor.is_root:
            ancestor = ancestor.parent_node
        return ancestor

    @staticmethod
    def get_expand_node(roots: List['TreeNode'], n: int = 1, mode: str = 'random') -> List['TreeNode']:
        """
        Sample n nodes from the pooled candidates of all trees in roots.
        """
        candidate_set = []
        for root in roots:
            candidate_set.extend(root.get_candidate_nodes())

        if mode == 'random':
            result = random.choices(candidate_set, k=n)
        elif mode == 'prob':
            # Plain top-n by log_sum_child_probs (logsumexp of the node's own children's
            # log_probs, -inf for a still-unexpanded node -- see _log_sum_child_probs). No
            # re-scoring after a pick: each of the n draws is a distinct node (the n lowest-key
            # candidates), not a repeated-selection priority queue. This was changed from an
            # earlier priority-queue design where a picked node's key only grew by ~log(number of
            # times picked) on each repeat -- too weak a penalty, so one clear outlier node could
            # (and empirically often did) win all n draws in a single iteration instead of
            # expansion spreading across n different nodes.
            raw_result = sorted(candidate_set, key=TreeNode._log_sum_child_probs)
            if len(raw_result) >= n:
                result = raw_result[:n]
            else:
                # Fewer distinct candidates than n -- pad with repeats of the single lowest-key
                # node rather than leaving result short (mirrors 'atpo' below).
                result = raw_result + [raw_result[0]] * (n - len(raw_result))
        elif mode == 'atpo':
            def f(node: 'TreeNode') -> float:
                return -TreeNode._min_child_log_prob(node) - 0.05 * (len(node.child_node) - 1)
            raw_result = sorted(candidate_set, key=f, reverse=True)
            if len(raw_result) >= n:
                result = raw_result[:n]
            else:
                # Fewer distinct candidates than n (e.g. a fresh root with no children yet) --
                # pad with repeats of the single highest-f node rather than leaving result short.
                result = raw_result + [raw_result[0]] * (n - len(raw_result))
        else:
            raise NotImplementedError

        assert len(result) == n, f"get_expand_node error, len(result)={len(result)} != n={n}"
        return result

    def sample_leaf(self, n: int = 1) -> List['TreeNode']:
        """
        Sample n leaves, then prune the tree (drop the unselected nodes)
        """

        candidate_uid_set = []

        for node in self.get_subtree_nodes():
            if node.is_leaf:
                candidate_uid_set.append(node.node_uid)

        if len(candidate_uid_set) < n:
            # dprint(f"root={self.node_uid}, candidate_uid_set={candidate_uid_set}")
            subtree_nodes = self.get_subtree_nodes()
            subtree_node_uids = [node.node_uid for node in subtree_nodes]
            # dprint(f"all subtree nodes={subtree_node_uids}")
        assert len(candidate_uid_set) >= n, f"root={self.node_uid}, candidate_uid_set len={len(candidate_uid_set)} < n={n}"

        random.shuffle(candidate_uid_set)
        # dprint(f'original candidate_uid_set len={len(candidate_uid_set)}')
        candidate_uid_set = candidate_uid_set[:n]
        # dprint(f'sampled candidate_uid_set len={len(candidate_uid_set)}')
        self._prune_subtree(candidate_uid_set)

        result = []
        for node in self.get_subtree_nodes():
            if node.is_leaf:
                result.append(node)
        return result

    def sample_all_leaves(self) -> List['TreeNode']:
        """
        Return every leaf in this tree, with no sampling or pruning. The total leaf count per
        prompt is deterministic from m/n/l (see compute_leaves_per_prompt), so unlike
        sample_leaf(k) (which prunes down to exactly k leaves), this just takes all of them.
        """
        result = []
        for node in self.get_subtree_nodes():
            if node.is_leaf:
                result.append(node)
        return result

    def set_leaf_original_score(self, score: float, status: Optional[str] = None):
        """
        Set the terminal (outcome) score of a leaf: correct / incorrect / invalid_submission /
        no_submission. status is the raw outcome label itself (see terminal_status), independent
        of whatever numeric score value the caller chose for it.
        """
        self.original_score = score
        self.terminal_status = status

    def set_node_additional_score(self, score: float):
        """
        Set the process-level score of this node, reflecting the single action that produced it
        from its parent (malformed_xml / invalid_arguments / unsafe_sql / sql_parse_error /
        sql_execution_error / sql_timeout, or 0. for a clean turn). May be called on leaf or
        non-leaf nodes.
        """
        self.additional_score = score

    @staticmethod
    def dfs_compute_utility(node: 'TreeNode') -> float:
        """
        Compute V(s) recursively bottom-up.
        V(leaf) = normalized_score
        V(non-leaf) = (1/m) * sum_i V(child_i)
        Also sets node.subtree_leaf_num = total leaf count in subtree.
        """
        if node.is_leaf or not node._child_node:
            node.utility = node.normalized_score
            node.subtree_leaf_num = 1
        else:
            child_utilities = []
            total_leaves = 0
            for child in node._child_node:
                TreeNode.dfs_compute_utility(child)
                child_utilities.append(child.utility)
                total_leaves += child.subtree_leaf_num
            node.utility = sum(child_utilities) / len(child_utilities)
            node.subtree_leaf_num = total_leaves
        return node.utility

    @staticmethod
    def dfs_compute_utility_leafmean(node: 'TreeNode') -> float:
        """
        Compute V(s) recursively bottom-up, for the 'tree_leafmean' reward_mode.
        V(leaf) = normalized_score
        V(non-leaf) = mean(normalized_score over every leaf in its subtree)
        Unlike dfs_compute_utility (equal weight per DIRECT CHILD, regardless of how many
        leaves are under it), this gives equal weight per LEAF -- a branch with more leaves
        pulls the average proportionally more. Also sets node.subtree_leaf_num = total leaf
        count in subtree (same meaning as dfs_compute_utility's).
        """
        if node.is_leaf or not node._child_node:
            node.utility = node.normalized_score
            node.subtree_leaf_num = 1
        else:
            leaf_score_sum = 0.0
            total_leaves = 0
            for child in node._child_node:
                TreeNode.dfs_compute_utility_leafmean(child)
                leaf_score_sum += child.utility * child.subtree_leaf_num
                total_leaves += child.subtree_leaf_num
            node.utility = leaf_score_sum / total_leaves
            node.subtree_leaf_num = total_leaves
        return node.utility

    @staticmethod
    def dfs_compute_reward_softmax(node: 'TreeNode') -> float:
        """
        Compute node.reward recursively bottom-up for the 'tree_softmax' reward_mode (used by
        the ATPO method).
        reward(leaf) is assumed to already be set to normalized_score by the caller before this
        is invoked from the root; this adds the node's own normalized_additional_score once.
        reward(non-leaf) = normalized_additional_score(node)
                            + sum_i softmax(child_i.log_prob over siblings) * reward(child_i)
        """
        if node.is_leaf or not node._child_node:
            node.reward = node.reward + node.normalized_additional_score
            return node.reward

        child_rewards = [TreeNode.dfs_compute_reward_softmax(child) for child in node._child_node]
        child_log_probs = torch.tensor([child.log_prob for child in node._child_node])
        weights = torch.softmax(child_log_probs, dim=0)
        node.reward = node.normalized_additional_score + sum(w.item() * r for w, r in zip(weights, child_rewards))
        return node.reward

    @staticmethod
    def normalize_and_compute_utility(roots: List['TreeNode']):
        """
        Center (subtract the mean, no division by std) each leaf's original_score into
        normalized_score across all leaves pooled from every tree in roots (the M trees
        belonging to the same prompt, i.e. sharing the same uid), then compute utility and
        subtree_leaf_num for every tree in roots.

        Also sets normalized_additional_score = additional_score (no normalization at all) for
        EVERY node (root + all descendants, not just leaves) in roots -- kept as a separate
        field/name from additional_score so every other method here can keep referring to
        "normalized_additional_score" unchanged.

        Note the 'grpo' adv_estimator (used by the grpo/gspo baselines) re-z-scores each
        sequence's summed score within the uid group anyway, so this function's scaling choice
        only actually reaches the final advantage for the 'tree'/'no_estimator' estimators
        (adtree/atpo), which use these values directly.

        Returns:
            'all_correct' | 'all_wrong' | 'mixed' | 'empty' -- this group's leaf original_score
            composition (correct is >0, wrong is <=0), for the caller to tally diagnostic
            group-composition counts. A group with neither (leaf_count == 0) reports 'empty'.
        """
        leaves = [node for root in roots for node in root.get_subtree_nodes() if node.is_leaf]
        leaf_count = len(leaves)
        scores = torch.tensor([node.original_score for node in leaves], dtype=torch.float32)

        centered = scores - scores.mean()
        for node, score in zip(leaves, centered.tolist()):
            node.normalized_score = score

        has_wrong = bool((scores <= 0).any().item()) if leaf_count > 0 else False
        has_correct = bool((scores > 0).any().item()) if leaf_count > 0 else False

        # 'tree_leafmean' uses the same reward formula as 'tree' (see
        # get_token_level_score_tree) but with utility as a flat mean over every leaf in the
        # subtree instead of a recursive mean over direct children -- see
        # dfs_compute_utility_leafmean. All roots here share one reward_mode (set uniformly at
        # construction from the run's config), so checking the first is enough.
        compute_utility_fn = (
            TreeNode.dfs_compute_utility_leafmean
            if roots and roots[0].reward_mode == 'tree_leafmean'
            else TreeNode.dfs_compute_utility
        )
        for root in roots:
            compute_utility_fn(root)

        for root in roots:
            for node in [root] + root.get_subtree_nodes():
                node.normalized_additional_score = node.additional_score

        if not has_wrong and not has_correct:
            return 'empty'
        if has_correct and has_wrong:
            return 'mixed'
        return 'all_correct' if has_correct else 'all_wrong'

    def get_token_level_score_from_leaf_with_process(self):
        """
        Token-level score for the 'base' reward_mode (used by the grpo/gspo baselines, which
        have no branching beyond m independent chains -- utility degenerates to the leaf's own
        normalized_score along a non-branching chain, so there is no benefit to the full tree
        utility/anchor formula there).

        Unlike get_token_level_score_tree / _tree_softmax, this places each turn's contribution
        at a SINGLE token (that turn's first token) rather than broadcasting it across the whole
        action's token span. 'base' feeds the 'grpo' advantage estimator, which sums
        token_level_rewards raw across the whole sequence (no leaf_count/step_first
        deduplication) -- broadcasting a turn's score across every one of its tokens would
        incorrectly scale that turn's contribution by how many tokens the action happened to
        take, which a per-sequence sum must not do.

        token_level_scores[first token of each turn's span] = normalized_additional_score(node)
        (walking from this leaf up to, but not including, root), PLUS normalized_score added at
        this leaf's own (final) turn's first token. All other tokens are 0.
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)
        final_token_level_leaf_counts = torch.ones_like(self.responses, dtype=torch.float32)
        final_token_level_step_first = torch.zeros_like(self.responses, dtype=torch.float32)

        valid_response_length_list = []
        r_list = []
        leaf_count_list = []

        node = self
        while node is not None and not node.is_root:
            valid_response_length = self.tensor_fn.create_attention_mask(node.responses).sum()
            valid_response_length_list.append(int(valid_response_length.item()))
            r = node.normalized_additional_score
            if node is self:
                r = r + node.normalized_score
            r_list.append(r)
            leaf_count_list.append(node.subtree_leaf_num)
            node = node.parent_node

        # Sentinel for the root boundary (length 0, no action)
        valid_response_length_list.append(0)
        r_list.append(0.0)
        leaf_count_list.append(1)

        # Reverse to go root -> leaf
        valid_response_length_list.reverse()
        r_list.reverse()
        leaf_count_list.reverse()

        for i in range(1, len(valid_response_length_list)):
            reward = r_list[i]
            lc = leaf_count_list[i]
            l = valid_response_length_list[i - 1]
            end = valid_response_length_list[i]  # exclusive upper bound (python slice semantics)
            if l < end:
                final_token_level_scores[l] = reward  # single point, NOT broadcast -- see docstring
                final_token_level_leaf_counts[l:end] = float(lc)
                final_token_level_step_first[l] = 1.0  # one representative token per action

        return final_token_level_scores, final_token_level_leaf_counts, final_token_level_step_first

    def get_token_level_score_tree(self):
        """
        Returns (token_level_scores, token_level_leaf_counts, token_level_step_first) for
        tree advantage estimation (used by the ADTree method's 'tree' reward_mode).

        token_level_scores[t] = r(a_k) = V(s_{k+1}) + gamma*(V(s_{k+1}) - V(anchor))
                                          + normalized_additional_score(s_{k+1})
                                 for tokens in action k's range, where anchor is the nearest
                                 ancestor of s_{k+1} with more than one child (the nearest actual
                                 branch point), or the root if none exists -- see _branch_anchor
        token_level_leaf_counts[t] = subtree_leaf_num(s_{k+1}) for those tokens
        token_level_step_first[t]  = 1.0 at the first token of each action's range, 0 elsewhere

        The step_first mask lets compute_advantage extract one (r, lc) pair per unique action and
        apply 1/lc importance-weighting to correctly deduplicate shared (non-leaf) nodes.

        Requires dfs_compute_utility and normalize_and_compute_utility to have been called from
        the root beforehand.
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)
        final_token_level_leaf_counts = torch.ones_like(self.responses, dtype=torch.float32)
        final_token_level_step_first = torch.zeros_like(self.responses, dtype=torch.float32)

        valid_response_length_list = []
        r_list = []
        leaf_count_list = []

        gamma = 0.4

        # Walk from leaf up to (but not including) root.
        # r(s) = V(s) + gamma*(V(s) - V(anchor)) + normalized_additional_score(s), where anchor
        # is the nearest ancestor of s with more than one child, or the root if none exists.
        node = self
        while node is not None and not node.is_root:
            valid_response_length = self.tensor_fn.create_attention_mask(node.responses).sum()
            valid_response_length_list.append(int(valid_response_length.item()))
            anchor = TreeNode._branch_anchor(node)
            r_list.append(
                node.utility + gamma * (node.utility - anchor.utility) + node.normalized_additional_score
            )
            leaf_count_list.append(node.subtree_leaf_num)
            node = node.parent_node

        # Sentinel for the root boundary (length 0, no action)
        valid_response_length_list.append(0)
        r_list.append(0.0)
        leaf_count_list.append(1)

        # Reverse to go root -> leaf
        valid_response_length_list.reverse()
        r_list.reverse()
        leaf_count_list.reverse()

        # Assign step reward and leaf count to each step's token range;
        # mark the first token of each range for action-level statistics in compute_advantage
        for i in range(1, len(valid_response_length_list)):
            reward = r_list[i]
            lc = leaf_count_list[i]
            l = valid_response_length_list[i - 1]
            end = valid_response_length_list[i]  # exclusive upper bound (python slice semantics)
            if l < end:
                final_token_level_scores[l:end] = reward
                final_token_level_leaf_counts[l:end] = float(lc)
                final_token_level_step_first[l] = 1.0  # one representative token per action

        return final_token_level_scores, final_token_level_leaf_counts, final_token_level_step_first

    def get_token_level_score_tree_softmax(self):
        """
        Returns (token_level_scores, token_level_leaf_counts, token_level_step_first) for the
        'tree_softmax' reward_mode (used by the ATPO method).

        token_level_scores[t] = node.reward for tokens in that node's own action range,
        walking from this leaf up to (but not including) the root. node.reward already includes
        each node's own normalized_additional_score (added in dfs_compute_reward_softmax).
        token_level_leaf_counts[t] = subtree_leaf_num(node) for those tokens
        token_level_step_first[t]  = 1.0 at the first token of each action's range, 0 elsewhere

        Requires dfs_compute_reward_softmax (and dfs_compute_utility, for subtree_leaf_num)
        to have been called from the root beforehand.
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)
        final_token_level_leaf_counts = torch.ones_like(self.responses, dtype=torch.float32)
        final_token_level_step_first = torch.zeros_like(self.responses, dtype=torch.float32)

        valid_response_length_list = []
        reward_list = []
        leaf_count_list = []

        node = self
        while node is not None and not node.is_root:
            valid_response_length = self.tensor_fn.create_attention_mask(node.responses).sum()
            valid_response_length_list.append(int(valid_response_length.item()))
            reward_list.append(node.reward)
            leaf_count_list.append(node.subtree_leaf_num)
            node = node.parent_node

        # Sentinel for the root boundary (length 0, no action)
        valid_response_length_list.append(0)
        reward_list.append(0.0)
        leaf_count_list.append(1)

        # Reverse to go root -> leaf
        valid_response_length_list.reverse()
        reward_list.reverse()
        leaf_count_list.reverse()

        for i in range(1, len(valid_response_length_list)):
            l = valid_response_length_list[i - 1]
            end = valid_response_length_list[i]  # exclusive upper bound (python slice semantics)
            lc = leaf_count_list[i]
            if l < end:
                final_token_level_scores[l:end] = reward_list[i]
                final_token_level_leaf_counts[l:end] = float(lc)
                final_token_level_step_first[l] = 1.0  # one representative token per action

        return final_token_level_scores, final_token_level_leaf_counts, final_token_level_step_first

    def _prune_subtree(self, candidate_uid_set: List[int]) -> bool:
        """
        Drop all the leaves not in cdandidate_uid_set
        """
        surviving_children = []
        # dprint(f'start, node={self.node_uid}, child_node len={len(self._child_node)}')

        for child in self.child_node:
            # dprint(f'node={self.node_uid}, iter for child={child.node_uid}')
            if child._prune_subtree(candidate_uid_set):
                surviving_children.append(child)

        # update child node
        self._child_node = surviving_children

        # Determine whether the current node should be retained. The conditions for retention are:
        # 1. It is in candidate_uid_set
        # 2. Or, it still has child nodes after pruning
        should_keep_this_node = self.node_uid in candidate_uid_set or (len(self._child_node) > 0)

        # dprint(f'end, node={self.node_uid}, is keeped={should_keep_this_node}')

        return should_keep_this_node

    def check_all_nodes_child(self):
        for node in self.get_subtree_nodes():
            node_child_list = node.child_node
            node_child_node_uid_list = []
            for node_child_node in node_child_list:
                node_child_node_uid_list.append(node_child_node.node_uid)
            if node.node_uid in node_child_node_uid_list:
                dprint(f'error!! node.uid in node_child_node_uid_list!!! is_root={node.is_root}, is_leaf={node.is_leaf}, node_uid={node.node_uid}')

    def delete_tree_from_root(self):
        """
        Delete the tree from root
        """
        all_nodes_list = self.get_subtree_nodes()
        all_nodes_list.append(self)

        for node in all_nodes_list:
            node.prompts = None
            node.input_ids = None
            node.attention_mask = None
            node.position_ids = None
            node.responses = None
            node.responses_with_info_mask = None

            node._child_node.clear()
            node.parent_node = None
