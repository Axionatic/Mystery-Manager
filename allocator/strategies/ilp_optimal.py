"""
ILP-optimal allocation strategy.

Integer Linear Programming via PuLP/CBC: minimise a composite penalty that
matches compare.py's scoring — convex piecewise-linear value penalty, 4D
diversity coverage, weighted dupe penalty, and MAD fairness proxy.

Falls back to deal-topup if PuLP is not installed or solver fails.
"""

import logging

from allocator.config import (
    BOX_TIERS,
    DIVERSITY_PENALTY_MULTIPLIER,
    DIVERSITY_WEIGHTS,
    DUPE_PENALTY_FLOOR,
    DUPE_PENALTY_MULTIPLIER,
    FAIRNESS_PENALTY_MULTIPLIER,
    ILP_BALANCE_WEIGHT,
    ILP_COVERAGE_WEIGHT,
    ILP_HHI_BREAKPOINTS,
    VALUE_ACCEPT_HIGH,
    VALUE_ACCEPT_LOW,
    VALUE_CEILING_PCT,
    VALUE_FAR_PENALTY_RATE,
    VALUE_HARD_HIGH,
    VALUE_NEAR_PENALTY_RATE,
    VALUE_OVER_FAR_RATE,
    VALUE_OVER_MODERATE_RATE,
    VALUE_SWEET_HIGH,
    VALUE_SWEET_LOW,
)
from allocator.models import AllocationResult
from allocator.strategies._helpers import compute_available_tags

logger = logging.getLogger(__name__)

TIME_LIMIT = 30  # seconds


def _compute_value_lines():
    """Derive piecewise-linear epigraph lines from config thresholds."""
    near_base = (VALUE_SWEET_LOW - VALUE_ACCEPT_LOW) * VALUE_NEAR_PENALTY_RATE
    over_soft_base = (VALUE_ACCEPT_HIGH - VALUE_SWEET_HIGH) * VALUE_NEAR_PENALTY_RATE
    over_hard_base = over_soft_base + (VALUE_HARD_HIGH - VALUE_ACCEPT_HIGH) * VALUE_OVER_MODERATE_RATE
    return [
        (-VALUE_FAR_PENALTY_RATE, VALUE_FAR_PENALTY_RATE * VALUE_ACCEPT_LOW + near_base),
        (-VALUE_NEAR_PENALTY_RATE, VALUE_NEAR_PENALTY_RATE * VALUE_SWEET_LOW),
        (0.0, 0.0),
        (VALUE_NEAR_PENALTY_RATE, -VALUE_NEAR_PENALTY_RATE * VALUE_SWEET_HIGH),
        (VALUE_OVER_MODERATE_RATE, -(VALUE_OVER_MODERATE_RATE * VALUE_ACCEPT_HIGH - over_soft_base)),
        (VALUE_OVER_FAR_RATE, -(VALUE_OVER_FAR_RATE * VALUE_HARD_HIGH - over_hard_base)),
    ]


_VALUE_LINES = _compute_value_lines()

# MAD-to-stddev scale factor: sqrt(pi/2) ≈ 1.2533
_MAD_STDDEV_FACTOR = 1.2533


def run(result: AllocationResult) -> None:
    """ILP allocation: optimal assignment via mixed-integer programming."""
    try:
        import pulp
    except ImportError:
        logger.warning("PuLP not installed, falling back to deal-topup")
        from allocator.strategies.deal_topup import run as fallback
        fallback(result)
        return

    if not result.boxes or not result.items:
        return

    try:
        _solve_ilp(result, pulp)
    except Exception as e:
        logger.warning(f"ILP solver failed ({e}), falling back to deal-topup")
        # Clear any partial allocations
        for box in result.boxes:
            box.allocations.clear()
        from allocator.strategies.deal_topup import run as fallback
        fallback(result)


def _solve_ilp(result: AllocationResult, pulp) -> None:
    """Build and solve the ILP model."""
    items = list(result.items.values())
    boxes = result.boxes
    item_ids = [i.id for i in items]
    n_items = len(items)
    n_boxes = len(boxes)

    prob = pulp.LpProblem("mystery_box_allocation", pulp.LpMinimize)

    # -----------------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------------

    # x[i][b] = integer qty of item i assigned to box b
    x = {}
    for i, item in enumerate(items):
        x[i] = {}
        for b in range(n_boxes):
            ub = item.overage
            if boxes[b].is_excluded(item):
                ub = 0
            x[i][b] = pulp.LpVariable(
                f"x_{item.id}_{b}", lowBound=0, upBound=ub, cat="Integer"
            )

    # y[i][b] = binary: is item i present in box b?
    y = {}
    for i, item in enumerate(items):
        y[i] = {}
        for b in range(n_boxes):
            y[i][b] = pulp.LpVariable(f"y_{item.id}_{b}", cat="Binary")

    # Link x and y: y[i][b] = 1 iff x[i][b] >= 1
    for i, item in enumerate(items):
        for b in range(n_boxes):
            prob += y[i][b] <= x[i][b]
            prob += x[i][b] <= item.overage * y[i][b]

    # -----------------------------------------------------------------------
    # Structural constraints
    # -----------------------------------------------------------------------

    # Overage: total assigned across all boxes <= overage
    for i, item in enumerate(items):
        prob += (
            pulp.lpSum(x[i][b] for b in range(n_boxes)) <= item.overage,
            f"overage_{item.id}",
        )

    # Ceiling: value in each box <= ceiling (% of box price)
    for b in range(n_boxes):
        box = boxes[b]
        prob += (
            pulp.lpSum(items[i].price * x[i][b] for i in range(n_items))
            <= VALUE_CEILING_PCT * BOX_TIERS[box.tier]["price"],
            f"ceiling_{b}",
        )

    # Hard fungible constraint: at most 1 item per fungible group per box
    # (for groups with degree >= 1.0)
    fungible_groups: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        if item.fungible_group and item.fungible_degree >= 1.0:
            fungible_groups.setdefault(item.fungible_group, []).append(i)

    for group_name, member_indices in fungible_groups.items():
        if len(member_indices) <= 1:
            continue
        for b in range(n_boxes):
            prob += (
                pulp.lpSum(y[mi][b] for mi in member_indices) <= 1,
                f"fungible_hard_{group_name}_{b}",
            )

    # -----------------------------------------------------------------------
    # 4a. Convex piecewise-linear value penalty
    # -----------------------------------------------------------------------
    # vp[b] = value as % of box price. Link: vp[b] * price_b == 100 * value_b
    vp = {}
    pen_val = {}
    for b in range(n_boxes):
        box = boxes[b]
        vp[b] = pulp.LpVariable(f"vp_{b}", lowBound=0)
        pen_val[b] = pulp.LpVariable(f"pen_val_{b}", lowBound=0)

        # Link vp to allocations: vp[b] = value / box_price * 100
        value_expr = pulp.lpSum(items[i].price * x[i][b] for i in range(n_items))
        box_price = BOX_TIERS[box.tier]["price"]
        prob += (
            box_price * vp[b] == 100 * value_expr,
            f"vp_link_{b}",
        )

        # Epigraph constraints: pen_val[b] >= slope * vp[b] + intercept
        for k, (slope, intercept) in enumerate(_VALUE_LINES):
            prob += (
                pen_val[b] >= slope * vp[b] + intercept,
                f"val_line_{b}_{k}",
            )

    # -----------------------------------------------------------------------
    # 4b. Diversity: binary coverage + HHI concentration penalty
    # -----------------------------------------------------------------------
    available_tags = compute_available_tags(result)

    # For each dimension, build tag→item-indices mapping
    dim_attrs = {
        "sub_category": "sub_category",
        "usage": "usage_type",
        "colour": "colour",
        "shape": "shape",
    }

    pen_div = {}
    for b in range(n_boxes):
        pen_div[b] = pulp.LpVariable(f"pen_div_{b}", lowBound=0)

    # --- Part (a): Binary coverage (same as before) ---
    coverage_exprs = {b: [] for b in range(n_boxes)}

    # Estimate total items per box (for HHI share normalisation)
    median_price = sorted(it.price for it in items)[len(items) // 2] if items else 1
    q_est = {b: max(boxes[b].target_value / max(median_price, 1), 1.0) for b in range(n_boxes)}

    # --- Part (b): HHI concentration penalty accumulators ---
    hhi_terms = {b: [] for b in range(n_boxes)}

    # Piecewise-linear tangent lines for f(s) = s^2 at breakpoints
    # At breakpoint p: tangent is f'(p)*(s - p) + p^2 = 2p*s - p^2
    tangent_lines = [(2 * p, -(p ** 2)) for p in ILP_HHI_BREAKPOINTS if p > 0]

    for dim, attr in dim_attrs.items():
        weight = DIVERSITY_WEIGHTS[dim]
        avail_tags = available_tags.get(dim, set())
        n_avail = len(avail_tags)

        if n_avail == 0:
            for b in range(n_boxes):
                coverage_exprs[b].append(weight)
            continue

        # Build tag → item indices
        tag_items: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            tag_val = getattr(item, attr, "")
            if tag_val and tag_val in avail_tags:
                tag_items.setdefault(tag_val, []).append(i)

        for tag in avail_tags:
            members = tag_items.get(tag, [])
            if not members:
                continue

            for b in range(n_boxes):
                # Binary coverage variable (unchanged)
                z_var = pulp.LpVariable(f"z_{dim}_{tag}_{b}", cat="Binary")
                prob += z_var <= pulp.lpSum(y[i][b] for i in members)
                for i in members:
                    prob += z_var >= y[i][b]
                coverage_exprs[b].append(weight / n_avail * z_var)

                # HHI: share = cnt / Q_est, sq >= share^2
                cnt = pulp.lpSum(x[i][b] for i in members)
                share = cnt / q_est[b]
                sq = pulp.LpVariable(f"sq_{dim}_{tag}_{b}", lowBound=0)
                # Tangent-line constraints: sq >= 2p*share - p^2
                for slope, intercept in tangent_lines:
                    prob += sq >= slope * share + intercept
                hhi_terms[b].append(sq)

    # Combined diversity penalty: alpha * DPM * (1 - coverage) + beta * DPM * hhi
    dpm = DIVERSITY_PENALTY_MULTIPLIER
    alpha = ILP_COVERAGE_WEIGHT
    beta = ILP_BALANCE_WEIGHT
    for b in range(n_boxes):
        coverage = pulp.lpSum(coverage_exprs[b])
        hhi_approx = pulp.lpSum(hhi_terms[b]) if hhi_terms[b] else 0
        prob += (
            pen_div[b] >= alpha * dpm * (1.0 - coverage) + beta * dpm * hhi_approx,
            f"div_pen_{b}",
        )

    # -----------------------------------------------------------------------
    # 4c. Soft dupe penalty
    # -----------------------------------------------------------------------
    # For each fungible group with effective weight > 0, model dupes
    all_fungible: dict[str, tuple[float, list[int]]] = {}
    for i, item in enumerate(items):
        if item.fungible_group:
            eff = max(item.fungible_degree - DUPE_PENALTY_FLOOR, 0.0)
            if item.fungible_group in all_fungible:
                _, members = all_fungible[item.fungible_group]
                members.append(i)
            else:
                all_fungible[item.fungible_group] = (eff, [i])

    pen_dupe = {}
    for b in range(n_boxes):
        pen_dupe[b] = pulp.LpVariable(f"pen_dupe_{b}", lowBound=0)

    dupe_terms = {b: [] for b in range(n_boxes)}
    for group_name, (eff_weight, member_indices) in all_fungible.items():
        if eff_weight <= 0 or len(member_indices) <= 1:
            continue

        for b in range(n_boxes):
            # count_in_box = sum of y[i][b] for members
            count_expr = pulp.lpSum(y[i][b] for i in member_indices)
            # dupe_var >= count - 1, dupe_var >= 0
            dv = pulp.LpVariable(f"dupe_{group_name}_{b}", lowBound=0)
            prob += dv >= count_expr - 1
            dupe_terms[b].append(eff_weight * DUPE_PENALTY_MULTIPLIER * dv)

    for b in range(n_boxes):
        if dupe_terms[b]:
            prob += pen_dupe[b] >= pulp.lpSum(dupe_terms[b]), f"dupe_pen_{b}"
        else:
            prob += pen_dupe[b] == 0, f"dupe_pen_{b}"

    # -----------------------------------------------------------------------
    # 4d. Fairness via MAD proxy
    # -----------------------------------------------------------------------
    # mean_vp = (1/n) * sum(vp[b]) — linear
    # abs_dev[b] >= |vp[b] - mean_vp| — 2 constraints each
    mean_vp_expr = pulp.lpSum(vp[b] for b in range(n_boxes)) / n_boxes

    abs_dev = {}
    for b in range(n_boxes):
        abs_dev[b] = pulp.LpVariable(f"abs_dev_{b}", lowBound=0)
        prob += abs_dev[b] >= vp[b] - mean_vp_expr, f"abs_dev_pos_{b}"
        prob += abs_dev[b] >= mean_vp_expr - vp[b], f"abs_dev_neg_{b}"

    # fairness_penalty = avg(abs_dev) * FAIRNESS_PENALTY_MULTIPLIER * MAD_STDDEV_FACTOR
    fair_pen = (
        pulp.lpSum(abs_dev[b] for b in range(n_boxes)) / n_boxes
        * FAIRNESS_PENALTY_MULTIPLIER * _MAD_STDDEV_FACTOR
    )

    # -----------------------------------------------------------------------
    # 4e. Objective: minimise avg(pen_val + pen_dupe + pen_div) + fair_pen
    # -----------------------------------------------------------------------
    avg_box_pen = pulp.lpSum(
        pen_val[b] + pen_dupe[b] + pen_div[b] for b in range(n_boxes)
    ) / n_boxes

    prob += avg_box_pen + fair_pen

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    try:
        solver = pulp.HiGHS(msg=0, timeLimit=TIME_LIMIT)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT)
    status = prob.solve(solver)

    if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
        if pulp.LpStatus[status] == "Infeasible":
            raise RuntimeError("ILP infeasible")

    if prob.sol_status not in (
        pulp.constants.LpSolutionOptimal,
        pulp.constants.LpSolutionIntegerFeasible,
    ):
        raise RuntimeError(f"No feasible solution found: {pulp.LpStatus[status]}")

    # -----------------------------------------------------------------------
    # Extract solution
    # -----------------------------------------------------------------------
    for i, item in enumerate(items):
        for b in range(n_boxes):
            val = x[i][b].varValue
            if val is not None and val > 0.5:
                qty = int(round(val))
                if qty > 0:
                    boxes[b].allocations[item.id] = (
                        boxes[b].allocations.get(item.id, 0) + qty
                    )

    total_assigned = sum(
        sum(q for q in box.allocations.values())
        for box in boxes
    )
    logger.info(
        f"ILP solved: status={pulp.LpStatus[status]}, "
        f"total items assigned={total_assigned}"
    )
