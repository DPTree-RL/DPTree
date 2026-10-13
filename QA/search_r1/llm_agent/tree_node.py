from __future__ import annotations

import math
from typing import List, Optional, Callable
from pydantic import BaseModel
import json
import random
import heapq
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
        reward_mode: str = 'base',
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

        self.original_score = 0.
        self.normalized_score = 0.
        self.weighted_normalized_score = 0.
        self.utility = 0.
        self.weighted_utility = 0.
        self.reward = 0.
        self.weight = 1.
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
            # Priority queue keyed by logsumexp of a per-node simulated logprobs list (starts
            # as the node's real children's log_probs). Each time a node is popped (selected),
            # append the logmeanexp of its current list back into the list and re-push with
            # the updated (necessarily higher, since logsumexp only grows when appending a
            # finite value) key -- simulating "if picked again, expect an average-ish new
            # branch" -- so a node can be selected more than once, but each repeat pick gets
            # progressively less attractive relative to the other candidates.
            heap = []
            for i, node in enumerate(candidate_set):
                logprobs = [child.log_prob for child in node.child_node]
                heapq.heappush(heap, (TreeNode._logsumexp(logprobs), i, node, logprobs))

            result = []
            counter = len(candidate_set)
            for _ in range(n):
                _, _, node, logprobs = heapq.heappop(heap)
                result.append(node)
                if logprobs:
                    log_mean_exp = TreeNode._logsumexp(logprobs) - math.log(len(logprobs))
                else:
                    log_mean_exp = 0.
                logprobs = logprobs + [log_mean_exp]
                heapq.heappush(heap, (TreeNode._logsumexp(logprobs), counter, node, logprobs))
                counter += 1
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

    def set_leaf_original_score(self, score: float):
        """
        Set the original score of the leaf
        """
        self.original_score = score

    @staticmethod
    def dfs_compute_weight(node: 'TreeNode'):
        """
        Compute node.weight recursively top-down: weight(root) = 1,
        weight(child) = weight(parent) / num_children(parent).
        Requires this to be called starting from the actual root.
        """
        if node.is_root:
            node.weight = 1.
        num_children = len(node._child_node)
        for child in node._child_node:
            child.weight = node.weight / num_children
            TreeNode.dfs_compute_weight(child)

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
    def dfs_compute_weighted_utility(node: 'TreeNode') -> float:
        """
        Same as dfs_compute_utility, but leaf scores use weighted_normalized_score instead of
        normalized_score. Sets node.weighted_utility (and node.subtree_leaf_num, same as
        dfs_compute_utility since it doesn't depend on which score variant is used).
        """
        if node.is_leaf or not node._child_node:
            node.weighted_utility = node.weighted_normalized_score
            node.subtree_leaf_num = 1
        else:
            child_utilities = []
            total_leaves = 0
            for child in node._child_node:
                TreeNode.dfs_compute_weighted_utility(child)
                child_utilities.append(child.weighted_utility)
                total_leaves += child.subtree_leaf_num
            node.weighted_utility = sum(child_utilities) / len(child_utilities)
            node.subtree_leaf_num = total_leaves
        return node.weighted_utility

    @staticmethod
    def dfs_compute_reward_softmax(node: 'TreeNode') -> float:
        """
        Compute node.reward recursively bottom-up for the 'tree_softmax' reward_mode.
        reward(leaf) is assumed to already be set (the leaf's normalized_score,
        set by the caller before this is invoked from the root).
        reward(non-leaf) = sum_i softmax(child_i.log_prob over siblings) * reward(child_i)
        """
        if node.is_leaf or not node._child_node:
            return node.reward

        child_rewards = [TreeNode.dfs_compute_reward_softmax(child) for child in node._child_node]
        child_log_probs = torch.tensor([child.log_prob for child in node._child_node])
        weights = torch.softmax(child_log_probs, dim=0)
        node.reward = sum(w.item() * r for w, r in zip(weights, child_rewards))
        return node.reward

    @staticmethod
    def normalize_and_compute_utility(roots: List['TreeNode'], epsilon: float = 1e-6):
        """
        Standardize (z-score) each leaf's original_score into normalized_score (unweighted) and
        weighted_normalized_score (weighted by node.weight), across all leaves pooled from every
        tree in roots (the M trees belonging to the same prompt, i.e. sharing the same uid;
        inter-tree), then compute utility (from normalized_score) and weighted_utility (from
        weighted_normalized_score), plus subtree_leaf_num, for every tree in roots. Every
        reward_mode and adv_estimator downstream sees only the standardized score -- none of
        them depend on the raw original_score value.
        """
        for root in roots:
            TreeNode.dfs_compute_weight(root)

        leaves = [node for root in roots for node in root.get_subtree_nodes() if node.is_leaf]
        scores = torch.tensor([node.original_score for node in leaves], dtype=torch.float32)

        normalized = (scores - scores.mean()) / (scores.std() + epsilon)
        for node, score in zip(leaves, normalized.tolist()):
            node.normalized_score = score

        w = torch.tensor([node.weight for node in leaves], dtype=torch.float32)
        total_w = w.sum()
        w_mean = (scores * w).sum() / total_w
        w_var = (w * (scores - w_mean) ** 2).sum() / total_w
        w_std = torch.sqrt(w_var)
        weighted_normalized = (scores - w_mean) / (w_std + epsilon)
        for node, score in zip(leaves, weighted_normalized.tolist()):
            node.weighted_normalized_score = score

        for root in roots:
            TreeNode.dfs_compute_utility(root)
            TreeNode.dfs_compute_weighted_utility(root)

    def get_token_level_score_from_leaf(self):
        """
        Get the token-level score from the leaf
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)

        valid_response_length = self.tensor_fn.create_attention_mask(self.responses).sum()
        final_token_level_scores[valid_response_length-1] = self.normalized_score

        return final_token_level_scores

    # @staticmethod
    # def _is_anchor_node(node: 'TreeNode', eps: float = 1e-6) -> bool:
    #     """An anchor node is one whose children's utilities are not all the same,
    #     i.e. at least one child's utility differs from this node's own utility
    #     (= the children's mean, per dfs_compute_utility) by more than eps.
    #     Special cases: a leaf is never an anchor; the root is always an anchor
    #     (checked first, so it wins even if the root happens to also be a leaf)."""
    #     if node.is_root:
    #         return True
    #     if node.is_leaf:
    #         return False
    #     return any(abs(child.utility - node.utility) > eps for child in node._child_node)

    @staticmethod
    def _find_anchor_ancestor(node: 'TreeNode', eps: float = 1e-6) -> 'TreeNode | None':
        """Return the nearest strict ancestor of node (starting from its parent)
        that is an anchor node. Since the root is always an anchor, this always
        finds one for any non-root node."""
        return node.parent_node
        # curr = node.parent_node
        # while curr is not None:
        #     if TreeNode._is_anchor_node(curr, eps):
        #         return curr
        #     curr = curr.parent_node
        # return None

    def get_token_level_score_tree(self):
        """
        Returns (token_level_scores, token_level_leaf_counts, token_level_step_first) for
        tree advantage estimation.

        token_level_scores[t]      = r(a_k) = V(s_{k+1}) - V(anchor) for tokens in action k's range
        token_level_leaf_counts[t] = subtree_leaf_num(s_{k+1}) for those tokens
        token_level_step_first[t]  = 1.0 at the first token of each action's range, 0 elsewhere

        The step_first mask lets compute_advantage extract one (r, lc) pair per unique action and
        apply 1/lc importance-weighting to correctly deduplicate shared (non-leaf) nodes.

        Requires dfs_compute_utility to have been called from the root beforehand.
        Unlike get_token_level_score_from_leaf, this method includes the leaf's own action step.
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)
        final_token_level_leaf_counts = torch.ones_like(self.responses, dtype=torch.float32)
        final_token_level_step_first = torch.zeros_like(self.responses, dtype=torch.float32)

        valid_response_length_list = []
        r_list = []
        leaf_count_list = []

        gamma = 0.4

        # Walk from leaf up to (but not including) root.
        # r(s) = V(s) - V(anchor), where anchor is the nearest strict ancestor of s
        # that is an anchor node (its children's utilities are not all the same;
        # the root is always an anchor, so this always resolves).
        node = self
        while node is not None and not node.is_root:
            valid_response_length = self.tensor_fn.create_attention_mask(node.responses).sum()
            valid_response_length_list.append(int(valid_response_length.item()))
            anchor_ancestor = TreeNode._find_anchor_ancestor(node)
            assert anchor_ancestor is not None, f"node_uid={node.node_uid} has no anchor ancestor (root should always be one)"
            r_list.append(node.utility + gamma * (node.utility - anchor_ancestor.utility))
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

    def get_token_level_score_tree_weighted(self):
        """
        Same as get_token_level_score_tree, but uses weighted_utility instead of utility, and
        scales the result by node.weight: r = weight * (V(s) - gamma * (V(s) - V(anchor))).
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)
        final_token_level_leaf_counts = torch.ones_like(self.responses, dtype=torch.float32)
        final_token_level_step_first = torch.zeros_like(self.responses, dtype=torch.float32)

        valid_response_length_list = []
        r_list = []
        leaf_count_list = []

        gamma = 0.4

        node = self
        while node is not None and not node.is_root:
            valid_response_length = self.tensor_fn.create_attention_mask(node.responses).sum()
            valid_response_length_list.append(int(valid_response_length.item()))
            anchor_ancestor = TreeNode._find_anchor_ancestor(node)
            assert anchor_ancestor is not None, f"node_uid={node.node_uid} has no anchor ancestor (root should always be one)"
            r_list.append(node.weight * (node.utility + gamma * (node.utility - anchor_ancestor.utility)))
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
                final_token_level_scores[l:end] = reward
                final_token_level_leaf_counts[l:end] = float(lc)
                final_token_level_step_first[l] = 1.0  # one representative token per action

        return final_token_level_scores, final_token_level_leaf_counts, final_token_level_step_first

    def get_token_level_score_tree_softmax(self):
        """
        Returns (token_level_scores, token_level_leaf_counts, token_level_step_first) for the
        'tree_softmax' reward_mode.

        token_level_scores[t] = node.reward for tokens in that node's own action range,
        walking from this leaf up to (but not including) the root. Unlike tree,
        this assigns each ancestor's own (softmax-weighted, bottom-up) reward directly,
        with no diff against a baseline.
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

