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


def greedy_action(obs: BloodObservation) -> DeliveryAction:
    """
    Priority order:
      1. If at hospital with needs and compatible blood in inventory → deliver
      2. If at blood source and inventory < 30% capacity → collect most-needed type
      3. Navigate toward nearest critical/high hospital (or blood source if low)
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

    # 1. Deliver if at hospital
    if current_zone and current_zone.zone_type == ZoneType.hospital:
        for need_type, need_qty in current_zone.needs.items():
            if need_qty <= 0:
                continue
            for donor_type in COMPATIBILITY.get(need_type, []):
                if inventory.get(donor_type, 0) > 0:
                    qty = min(inventory[donor_type], need_qty, 20)
                    return DeliveryAction(
                        action_type=ActionType.deliver,
                        target_zone_id=agent.current_zone_id,
                        blood_type=donor_type,
                        quantity=qty,
                    )

    low_inventory = inv_total < capacity * 0.30

    # 2. Collect if at source
    if current_zone and current_zone.zone_type in (ZoneType.blood_bank, ZoneType.donor_center):
        if low_inventory or cap_remaining > 20:
            # Pick blood type most needed globally
            need_counts: dict = {}
            for z in obs.zones:
                if z.zone_type == ZoneType.hospital:
                    for bt, qty in z.needs.items():
                        need_counts[bt] = need_counts.get(bt, 0) + qty

            best_bt, best_score_val = None, -1
            for bt, stock_qty in current_zone.stock.items():
                if stock_qty > 0:
                    s = need_counts.get(bt, 0)
                    if s > best_score_val:
                        best_score_val, best_bt = s, bt
            if best_bt is None:
                for bt in BLOOD_TYPES:
                    if current_zone.stock.get(bt, 0) > 0:
                        best_bt = bt
                        break
            if best_bt:
                qty = min(cap_remaining, current_zone.stock.get(best_bt, 0), 30)
                if qty > 0:
                    return DeliveryAction(
                        action_type=ActionType.collect,
                        target_zone_id=agent.current_zone_id,
                        blood_type=best_bt,
                        quantity=qty,
                    )

    # 3. Navigate
    if low_inventory:
        candidates = [
            z for z in obs.zones
            if z.zone_type in (ZoneType.blood_bank, ZoneType.donor_center)
            and sum(z.stock.values()) > 0
        ]
        target = min(candidates, key=lambda z: abs(z.x - ax) + abs(z.y - ay)) \
            if candidates else None
    else:
        priority = sorted(
            [z for z in obs.zones
             if z.zone_type == ZoneType.hospital and sum(z.needs.values()) > 0],
            key=lambda z: (
                {"critical": 0, "high": 1, "moderate": 2, "low": 3, "stable": 4}.get(
                    z.urgency.value, 4),
                abs(z.x - ax) + abs(z.y - ay),
            )
        )
        target = priority[0] if priority else None

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
    # Clamp strictly within (0, 1) — not 0.0 and not 1.0 — as required by the evaluator
    return round(max(0.01, min(0.99, score)), 4)

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
    lives_pct = st.get("lives_saved_pct", 0.01)
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
