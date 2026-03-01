"""
Pluggable allocation strategies.

A strategy is a callable (AllocationResult) -> None that fills box.allocations
in place. Everything before (data loading, box building) and after (CCI
allocation, stock) is shared infrastructure in allocate().
"""

from typing import Callable

from allocator.models import AllocationResult

Strategy = Callable[[AllocationResult], None]

_REGISTRY: dict[str, tuple[str, str]] = {
    # name -> (module_path, function_name) — lazy-loaded to avoid circular imports
    "deal-topup": ("allocator.strategies.deal_topup", "run"),
    "greedy-best-fit": ("allocator.strategies.greedy_best_fit", "run"),
    "round-robin": ("allocator.strategies.round_robin", "run"),
    "minmax-deficit": ("allocator.strategies.minmax_deficit", "run"),
    "local-search": ("allocator.strategies.local_search", "run"),
    "ilp-optimal": ("allocator.strategies.ilp_optimal", "run"),
    "discard-worst": ("allocator.strategies.discard_worst", "run"),
}

DEFAULT_STRATEGY = "deal-topup"


def get_strategy(name: str) -> Strategy:
    """Look up a strategy by name, importing its module lazily."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name!r}. Available: {available}")
    module_path, func_name = _REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def list_strategies() -> list[str]:
    return list(_REGISTRY.keys())
