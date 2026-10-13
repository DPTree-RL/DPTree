def compute_leaves_per_prompt(m: int, n: int, l: int) -> int:
    """
    Deterministic number of final leaves per original prompt after l rounds of tree-search
    expansion, given m initial chains and n nodes expanded per round.

    Expansion always picks candidates from non-leaf nodes only (see
    TreeNode.get_candidate_nodes), so every pick attaches one new child leaf without turning
    any existing leaf into a non-leaf. The total leaf count therefore doesn't depend on which
    specific nodes get picked, only on m, n, l.

    All m initial chains are grown from a single shared root per prompt (see
    LLMGenerationTreeSearchManager.run_llm_loop_tree_search), so every expand_mode -- whether
    it pools candidates across the whole tree ('inter_*') or picks per-tree ('random'/'prob',
    which now means "the one tree of this prompt") -- selects n new leaves per round out of
    the same pool:
      -> m + l * n
    """
    return m + l * n
