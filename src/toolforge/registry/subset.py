"""Stratified subset filter (F1.4).

select_subset() reduces a full normalized tool list to a target endpoint
budget, stratified across all categories present.  Deterministic given the
same (tools, target_endpoints, seed) triple.

No LLM calls.  Pure deterministic Python.
"""

from __future__ import annotations

import random
from collections import defaultdict

import structlog

from toolforge.registry.models import Tool

log = structlog.get_logger(__name__)

_MAX_ENDPOINTS_PER_TOOL = 10
_MIN_TOOLS_PER_CATEGORY = 3


def select_subset(
    tools: list[Tool],
    target_endpoints: int = 500,
    seed: int = 42,
) -> list[Tool]:
    """Return a deterministic stratified subset of *tools*.

    Algorithm
    ---------
    1. Drop tools where the tool description is empty AND every endpoint
       description is also empty (nothing to ground conversations in).
    2. Drop endpoints with zero parameters; drop tools left with none.
    3. Cap each tool at ``_MAX_ENDPOINTS_PER_TOOL`` (10) endpoints, taking
       the first N after quality filtering.  This prevents a single large
       tool from consuming an entire category budget.
    4. Stratify across categories: allocate ``target_endpoints // n_cats``
       to each (remainder distributed round-robin).
    5. Within each category, rank tools: ≥1 required param first; ties
       broken by endpoint count descending, then tool name ascending.
    6. Greedily add whole tools until the category endpoint budget is met.
    7. Enforce ``_MIN_TOOLS_PER_CATEGORY`` (3): if fewer tools were selected,
       pull additional tools from the ranked list even if it slightly exceeds
       the budget.
    """
    rng = random.Random(seed)

    # --- Step 1: drop fully undescribed tools ---
    described: list[Tool] = []
    drop_undescribed = 0
    for t in tools:
        if not t.description.strip() and all(
            not ep.description.strip() for ep in t.endpoints
        ):
            drop_undescribed += 1
            continue
        described.append(t)
    if drop_undescribed:
        log.info("subset.dropped_undescribed", count=drop_undescribed)

    # --- Step 2: drop zero-param endpoints; drop tools left with none ---
    pruned: list[Tool] = []
    drop_no_params = 0
    ep_pruned = 0
    for t in described:
        kept_eps = tuple(ep for ep in t.endpoints if ep.parameters)
        if not kept_eps:
            drop_no_params += 1
            continue
        if len(kept_eps) < len(t.endpoints):
            ep_pruned += len(t.endpoints) - len(kept_eps)
            t = t.model_copy(update={"endpoints": kept_eps})
        pruned.append(t)
    if drop_no_params:
        log.info("subset.dropped_no_params_tools", count=drop_no_params)
    if ep_pruned:
        log.info("subset.pruned_zero_param_endpoints", count=ep_pruned)

    # --- Step 3: cap each tool at _MAX_ENDPOINTS_PER_TOOL ---
    capped: list[Tool] = []
    ep_capped = 0
    for t in pruned:
        if len(t.endpoints) > _MAX_ENDPOINTS_PER_TOOL:
            ep_capped += len(t.endpoints) - _MAX_ENDPOINTS_PER_TOOL
            t = t.model_copy(
                update={"endpoints": t.endpoints[:_MAX_ENDPOINTS_PER_TOOL]}
            )
        capped.append(t)
    if ep_capped:
        log.info("subset.capped_endpoints", count=ep_capped)

    # --- Step 4: stratify budget across categories ---
    by_category: dict[str, list[Tool]] = defaultdict(list)
    for t in capped:
        by_category[t.category].append(t)

    categories = sorted(by_category)
    n_cats = len(categories)
    if n_cats == 0:
        return []

    base_budget = target_endpoints // n_cats
    remainder = target_endpoints - base_budget * n_cats
    budgets: dict[str, int] = {
        cat: base_budget + (1 if i < remainder else 0)
        for i, cat in enumerate(categories)
    }

    # --- Steps 5–7: rank, greedily select, enforce minimum tool count ---
    selected: list[Tool] = []

    for cat in categories:
        cat_tools = by_category[cat]
        budget = budgets[cat]

        def _rank_key(t: Tool) -> tuple[int, int, str]:
            has_required = any(
                p.required for ep in t.endpoints for p in ep.parameters
            )
            return (0 if has_required else 1, -len(t.endpoints), t.name)

        # Deterministic shuffle then stable re-sort so equal-ranked tools
        # don't follow filesystem order.
        shuffled = list(cat_tools)
        rng.shuffle(shuffled)
        ranked = sorted(shuffled, key=_rank_key)

        ep_count = 0
        cat_selected: list[Tool] = []
        remaining: list[Tool] = []

        for t in ranked:
            if ep_count + len(t.endpoints) <= budget:
                cat_selected.append(t)
                ep_count += len(t.endpoints)
            else:
                remaining.append(t)

        # Step 7: enforce minimum tool count — pull from remaining even if
        # it slightly exceeds the endpoint budget.
        if len(cat_selected) < _MIN_TOOLS_PER_CATEGORY:
            needed = _MIN_TOOLS_PER_CATEGORY - len(cat_selected)
            for t in remaining[:needed]:
                cat_selected.append(t)
                ep_count += len(t.endpoints)

        log.info(
            "subset.category_selected",
            category=cat,
            tools=len(cat_selected),
            endpoints=ep_count,
            budget=budget,
        )
        selected.extend(cat_selected)

    return selected
