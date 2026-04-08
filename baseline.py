"""
baseline.py – Greedy baseline agent for the Blood Bank Supply environment.

Runs all three scenarios using a deterministic greedy heuristic and emits
the required structured stdout log format:
  [START] {...}
  [STEP]  {...}
  [END]   {...}

Usage:
    python baseline.py

No API keys required. Results are fully reproducible given the same RNG_SEED.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import deque
from typing import Optional

from dotenv import load_dotenv

from environment import (
    BloodBankEnvironment,
    BloodObservation,
    DeliveryAction,
    ActionType,
    Direction,
    ZoneType,
    BLOOD_TYPES,
    COMPATIBILITY,
    SCENARIOS,
)

load_dotenv()

RNG_SEED: int = int(os.environ.get("RNG_SEED", "42"))

TASKS = [
    ("easy",   "city_shortage"),
    ("medium", "rare_type_emergency"),
    ("hard",   "disaster_response"),
]

# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------

def log_start(task_id: str, scenario: str, seed: int) -> None:
    sys.stdout.write(
        f"[START] {{\"task_id\": \"{task_id}\", \"scenario\": \"{scenario}\", \"seed\": {seed}}}\n"
    )
    sys.stdout.flush()


def log_step(step: int, action: dict, reward: float, obs: BloodObservation) -> None:
    payload = {
        "step": step,
        "action": action,
        "reward": round(reward, 4),
        "lives_saved_pct": obs.lives_saved_pct,
        "patients_saved": obs.patients_saved,
        "patients_lost": obs.patients_lost,
        "is_complete": obs.is_complete,
    }
    sys.stdout.write(f"[STEP] {json.dumps(payload)}\n")
    sys.stdout.flush()


def log_end(task_id: str, score: float, lives_pct: float, steps: int,
            success: bool) -> None:
    payload = {
        "task_id": task_id,
        "score": round(score, 4),
        "lives_saved_pct": round(lives_pct, 2),
        "steps": steps,
        "mission_success": success,
    }
    sys.stdout.write(f"[END] {json.dumps(payload)}\n")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Greedy policy
# ---------------------------------------------------------------------------

def _bfs_direction(ax: int, ay: int, tx: int, ty: int,
                   blocked: set) -> Optional[Direction]:
    """BFS on 10×10 grid; returns first direction toward (tx, ty)."""
    if ax == tx and ay == ty:
        return None
    dir_map = {
        Direction.north: (0, -1),
        Direction.south: (0, 1),
        Direction.west:  (-1, 0),
        Direction.east:  (1, 0),
    }
    queue: deque = deque([(ax, ay, [])])
    visited = {(ax, ay)}
    while queue:
        cx, cy, path = queue.popleft()
        for d, (ddx, ddy) in dir_map.items():
            nx, ny = cx + ddx, cy + ddy
            if not (0 <= nx < 10 and 0 <= ny < 10):
                continue
            if (nx, ny) in visited or (nx, ny) in blocked:
                continue
            new_path = path + [d]
            if nx == tx and ny == ty:
                return new_path[0]
            visited.add((nx, ny))
            queue.append((nx, ny, new_path))
    # Greedy fallback
    if tx > ax:
        return Direction.east
    if tx < ax:
        return Direction.west
    if ty > ay:
        return Direction.south
    return Direction.north


def _bfs_dist(ax: int, ay: int, tx: int, ty: int, blocked: set) -> int:
    """BFS distance between two points, respecting blocked cells."""
    if ax == tx and ay == ty:
        return 0
    queue: deque = deque([(ax, ay, 0)])
    visited = {(ax, ay)}
    while queue:
        cx, cy, d = queue.popleft()
        for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + ddx, cy + ddy
            if not (0 <= nx < 10 and 0 <= ny < 10):
                continue
            if (nx, ny) in visited or (nx, ny) in blocked:
                continue
            if nx == tx and ny == ty:
                return d + 1
            visited.add((nx, ny))
            queue.append((nx, ny, d + 1))
    return 999  # unreachable


def _donor_usefulness(obs: BloodObservation) -> dict:
    """How many hospital units each donor blood type can satisfy."""
    usefulness: dict = {}
    for z in obs.zones:
        if z.zone_type != ZoneType.hospital:
            continue
        for need_type, need_qty in z.needs.items():
            if need_qty <= 0:
                continue
            for donor_type in COMPATIBILITY.get(need_type, []):
                usefulness[donor_type] = usefulness.get(donor_type, 0) + need_qty
    return usefulness


def greedy_action(obs: BloodObservation) -> DeliveryAction:
    """
    Improved greedy policy:
      1. Deliver at hospital — all compatible types, up to 50 units per step.
      2. Collect at blood source — most-useful type first, up to 50 per step,
         while low on inventory OR there is capacity remaining.
      3. Navigate:
         - If inventory >= 20% capacity: go to most urgent deliverable hospital
           (BFS dist + dying bonus for steps_unserved >= threshold).
         - Otherwise: go to nearest blood source.
    """
    agent = obs.agent
    ax, ay = agent.x, agent.y
    inventory = agent.inventory
    inv_total = agent.total_units
    cap_remaining = agent.capacity_remaining
    capacity = inv_total + cap_remaining

    zone_map = {z.zone_id: z for z in obs.zones}
    current_zone = zone_map.get(agent.current_zone_id)
    blocked = {(z.x, z.y) for z in obs.zones if z.zone_type == ZoneType.blocked}

    # 1. Deliver if at hospital — sort needs by urgency-weighted quantity
    if current_zone and current_zone.zone_type == ZoneType.hospital:
        for need_type, need_qty in sorted(current_zone.needs.items(), key=lambda x: -x[1]):
            if need_qty <= 0:
                continue
            for donor_type in COMPATIBILITY.get(need_type, []):
                if inventory.get(donor_type, 0) > 0:
                    qty = min(inventory[donor_type], need_qty, 50)
                    return DeliveryAction(
                        action_type=ActionType.deliver,
                        target_zone_id=agent.current_zone_id,
                        blood_type=donor_type,
                        quantity=qty,
                    )

    low_inventory = inv_total < capacity * 0.20

    # 2. Collect if at blood source and worthwhile
    if current_zone and current_zone.zone_type in (ZoneType.blood_bank, ZoneType.donor_center):
        if low_inventory or cap_remaining > 10:
            usefulness = _donor_usefulness(obs)
            best_bt, best_val = None, -1.0
            for bt in BLOOD_TYPES:
                stock = current_zone.stock.get(bt, 0)
                if stock <= 0:
                    continue
                # Prefer types with high demand relative to what we already carry
                val = usefulness.get(bt, 0) - inventory.get(bt, 0) * 0.5
                if val > best_val:
                    best_val = val
                    best_bt = bt
            if best_bt is None:
                for bt in BLOOD_TYPES:
                    if current_zone.stock.get(bt, 0) > 0:
                        best_bt = bt
                        break
            if best_bt:
                qty = min(cap_remaining, current_zone.stock.get(best_bt, 0), 50)
                if qty > 0:
                    return DeliveryAction(
                        action_type=ActionType.collect,
                        target_zone_id=agent.current_zone_id,
                        blood_type=best_bt,
                        quantity=qty,
                    )

    # 3. Navigate
    urgency_rank = {"critical": 0, "high": 1, "moderate": 2, "low": 3, "stable": 4}

    if not low_inventory:
        # Head to the most urgent hospital — prefer ones where we have compatible blood
        hospitals = [
            z for z in obs.zones
            if z.zone_type == ZoneType.hospital and sum(z.needs.values()) > 0
        ]
        # Separate deliverable vs non-deliverable (go to deliverable first)
        def _can_deliver(z: object) -> bool:
            for nt, nq in z.needs.items():  # type: ignore[attr-defined]
                if nq <= 0:
                    continue
                for dt in COMPATIBILITY.get(nt, []):
                    if inventory.get(dt, 0) > 0:
                        return True
            return False

        deliverable = [z for z in hospitals if _can_deliver(z)]
        pool = deliverable if deliverable else hospitals

        def _hospital_key(z: object) -> tuple:
            dist = _bfs_dist(ax, ay, z.x, z.y, blocked)  # type: ignore[attr-defined]
            urank = urgency_rank.get(z.urgency.value, 4)  # type: ignore[attr-defined]
            # Dying: critical ≥ 3 steps or high ≥ 5 steps unserved → top priority
            dying = int(
                (z.urgency.value == "critical" and z.steps_unserved >= 3) or  # type: ignore[attr-defined]
                (z.urgency.value == "high" and z.steps_unserved >= 5)  # type: ignore[attr-defined]
            )
            return (1 - dying, urank, dist)

        target = min(pool, key=_hospital_key) if pool else None
    else:
        # Low inventory — go to nearest blood source with stock
        candidates = [
            z for z in obs.zones
            if z.zone_type in (ZoneType.blood_bank, ZoneType.donor_center)
            and sum(z.stock.values()) > 0
        ]
        target = min(candidates,
                     key=lambda z: _bfs_dist(ax, ay, z.x, z.y, blocked)) \
            if candidates else None

    if target is None:
        return DeliveryAction(action_type=ActionType.wait)

    direction = _bfs_direction(ax, ay, target.x, target.y, blocked)
    if direction:
        return DeliveryAction(action_type=ActionType.move, direction=direction)

    return DeliveryAction(action_type=ActionType.wait)

# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

def _compute_score(lives_pct: float, steps_used: int, max_steps: int,
                   capacity: int, cap_remaining: int) -> float:
    utilization = max(0.0, 1.0 - cap_remaining / capacity) if capacity > 0 else 0.0
    speed = max(0.0, 1.0 - steps_used / max_steps) if max_steps > 0 else 0.0
    score = 0.7 * (lives_pct / 100.0) + 0.15 * utilization + 0.15 * speed
    return max(0.0001, min(0.9999, round(score, 4)))

# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

async def run_task(task_id: str, scenario_name: str, seed: int) -> dict:
    log_start(task_id, scenario_name, seed)

    env = BloodBankEnvironment(scenario_name, rng_seed=seed)
    obs = await env.reset()
    done = False
    step_count = 0

    while not done:
        action = greedy_action(obs)
        action_dict = json.loads(action.model_dump_json())
        obs, reward, done, _ = await env.step(action)
        step_count += 1
        log_step(step_count, action_dict, reward, obs)

    st = env.state
    capacity = SCENARIOS.get(scenario_name, {}).get("capacity", 100)
    cap_remaining = st.get("capacity_remaining", capacity)
    lives_pct = st.get("lives_saved_pct", 0.0)
    score = _compute_score(lives_pct, step_count, obs.max_steps, capacity, cap_remaining)
    success = bool(st.get("mission_success", False))

    log_end(task_id, score, lives_pct, step_count, success)

    return {
        "task_id": task_id,
        "scenario": scenario_name,
        "score": score,
        "lives_saved_pct": round(lives_pct, 2),
        "steps_used": step_count,
        "mission_success": success,
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    results = []
    for task_id, scenario_name in TASKS:
        result = await run_task(task_id, scenario_name, RNG_SEED)
        results.append(result)
        print(
            f"[INFO] {task_id}: score={result['score']}  "
            f"lives={result['lives_saved_pct']}%  success={result['mission_success']}",
            file=sys.stderr,
        )

    avg_score = round(sum(r["score"] for r in results) / len(results), 4)
    print(f"\n[SUMMARY] avg_score={avg_score}", file=sys.stderr)
    print(json.dumps({"summary": results, "avg_score": avg_score}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
