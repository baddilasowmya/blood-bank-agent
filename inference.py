"""
inference.py – LLM-driven agent for the Blood Bank Supply environment.

Runs all three scenarios (easy / medium / hard) using an LLM to decide
each action, and emits the required structured stdout log format:
  [START] {...}
  [STEP]  {...}
  [END]   {...}

Usage:
    python inference.py

Required environment variables:
    API_BASE_URL  – OpenAI-compatible endpoint (provided by hackathon proxy)
    MODEL_NAME    – Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      – Bearer token for the API
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import re
from collections import deque
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.environ.get("HF_TOKEN",     "")
RNG_SEED: int     = int(os.environ.get("RNG_SEED",  "42"))

TASKS = [
    ("easy",   "city_shortage"),
    ("medium", "rare_type_emergency"),
    ("hard",   "disaster_response"),
]

# ---------------------------------------------------------------------------
# LLM client  (uses API_BASE_URL from env — hackathon proxy or HF router)
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)

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
        "reward": max(0.0001, min(0.9999, round(reward, 4))),
        "lives_saved_pct": max(0.0001, min(0.9999, round(obs.lives_saved_pct / 100.0, 4))),
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
        "score": max(0.0001, min(0.9999, round(score, 4))),
        "lives_saved_pct": max(0.0001, min(0.9999, round(lives_pct / 100.0, 4))),
        "steps": steps,
        "mission_success": success,
    }
    sys.stdout.write(f"[END] {json.dumps(payload)}\n")
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Observation → prompt
# ---------------------------------------------------------------------------

def _obs_to_prompt(obs: BloodObservation) -> str:
    agent = obs.agent
    lines = [
        f"=== Blood Bank Supply Agent – Step {obs.step_number}/{obs.max_steps} ===",
        f"Scenario: {obs.scenario_name}",
        f"Agent position: ({agent.x}, {agent.y})  zone={agent.current_zone_id}",
        f"Inventory ({agent.total_units}/{agent.total_units + agent.capacity_remaining}): "
        + ", ".join(f"{bt}:{agent.inventory.get(bt, 0)}"
                    for bt in BLOOD_TYPES if agent.inventory.get(bt, 0) > 0),
        f"Patients – total:{obs.total_patients}  saved:{obs.patients_saved}"
        f"  lost:{obs.patients_lost}  ({obs.lives_saved_pct:.1f}% saved)",
        "",
        "HOSPITALS (unserved only):",
    ]

    for z in sorted(
        [z for z in obs.zones
         if z.zone_type == ZoneType.hospital and sum(z.needs.values()) > 0],
        key=lambda z: ({"critical": 0, "high": 1, "moderate": 2,
                        "low": 3, "stable": 4}.get(z.urgency.value, 4),
                       abs(z.x - agent.x) + abs(z.y - agent.y)),
    ):
        needs_str = ", ".join(f"{bt}:{qty}" for bt, qty in z.needs.items() if qty > 0)
        lines.append(
            f"  [{z.urgency.value.upper()}] {z.name} ({z.zone_id}) at ({z.x},{z.y})"
            f"  needs: {needs_str}  waiting:{z.patients_waiting}"
            f"  unserved_steps:{z.steps_unserved}"
        )

    lines += ["", "BLOOD SOURCES:"]
    for z in obs.zones:
        if z.zone_type in (ZoneType.blood_bank, ZoneType.donor_center):
            stock_str = ", ".join(f"{bt}:{qty}"
                                  for bt, qty in z.stock.items() if qty > 0)
            if stock_str:
                lines.append(
                    f"  {z.name} ({z.zone_id}) at ({z.x},{z.y})  stock: {stock_str}"
                )

    lines += [
        "",
        "Last action result: " + obs.last_action_result,
        "",
        "BLOOD COMPATIBILITY (patient_need: compatible_donors):",
        "  O-: [O-]   O+: [O+,O-]   A-: [A-,O-]   A+: [A+,A-,O+,O-]",
        "  B-: [B-,O-]  B+: [B+,B-,O+,O-]  AB-: [AB-,A-,B-,O-]  AB+: [ANY]",
        "  KEY: O- blood can be given to ANY patient. Prioritise collecting O-.",
        "",
        "INSTRUCTIONS:",
        "Choose ONE action. Reply with ONLY a valid JSON object (no markdown, no explanation).",
        "Valid action_types: move | deliver | collect | wait",
        "  move:    {\"action_type\": \"move\", \"direction\": \"north|south|east|west\"}",
        "  deliver: {\"action_type\": \"deliver\", \"target_zone_id\": \"Z_x_y\","
        " \"blood_type\": \"O+\", \"quantity\": 50}",
        "  collect: {\"action_type\": \"collect\", \"target_zone_id\": \"Z_x_y\","
        " \"blood_type\": \"O+\", \"quantity\": 50}",
        "  wait:    {\"action_type\": \"wait\"}",
        "",
        "STRATEGY (follow in order):",
        "1. DELIVER: If at a hospital with needs and you have compatible blood → deliver immediately.",
        "   - Check compatibility table: e.g. to satisfy O+ need, use O+ or O- blood.",
        "   - Deliver up to 50 units per action. Stay and deliver multiple types if needed.",
        "2. COLLECT: If at a blood source and capacity_remaining > 0 → collect most-needed type.",
        "   - Fill up completely before leaving. Prefer O- (universal), then O+, then specific types.",
        "3. MOVE to CRITICAL/HIGH hospital (unserved_steps ≥ 2) if you have compatible blood.",
        "   - CRITICAL hospitals die after 3 unserved steps (5%/step). RUSH THERE.",
        "   - HIGH hospitals die after 5 unserved steps (3%/step). High priority.",
        "4. MOVE to nearest blood source if inventory < 20% OR no hospital can be helped.",
        "NEVER wait unless no other option. Every step matters!",
    ]

    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are an expert logistics AI managing blood bank supply chains in a life-or-death emergency. "
    "Your ONLY goal is to save ≥85% of patients by delivering the right blood type to hospitals "
    "before patients die. CRITICAL hospitals lose patients after 3 unserved steps; HIGH after 5. "
    "O- blood is universal — it can be given to ANY patient type. "
    "Always reply with a single valid JSON action object and NOTHING else. No explanation, no markdown."
)

# ---------------------------------------------------------------------------
# Greedy fallback (used when LLM call fails)
# ---------------------------------------------------------------------------

def _bfs_direction(ax: int, ay: int, tx: int, ty: int,
                   blocked: set) -> Optional[Direction]:
    if ax == tx and ay == ty:
        return None
    dir_map = {
        Direction.north: (0, -1), Direction.south: (0, 1),
        Direction.west:  (-1, 0), Direction.east:  (1, 0),
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
    if tx > ax: return Direction.east
    if tx < ax: return Direction.west
    if ty > ay: return Direction.south
    return Direction.north


def _bfs_dist(ax: int, ay: int, tx: int, ty: int, blocked: set) -> int:
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
    return 999


def _fallback_action(obs: BloodObservation) -> DeliveryAction:
    """Improved greedy fallback used when LLM call fails."""
    agent = obs.agent
    ax, ay = agent.x, agent.y
    inventory = agent.inventory
    inv_total = agent.total_units
    cap_remaining = agent.capacity_remaining
    capacity = inv_total + cap_remaining

    zone_map = {z.zone_id: z for z in obs.zones}
    current_zone = zone_map.get(agent.current_zone_id)
    blocked = {(z.x, z.y) for z in obs.zones if z.zone_type == ZoneType.blocked}

    # 1. Deliver at hospital
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

    # 2. Collect at blood source
    if current_zone and current_zone.zone_type in (ZoneType.blood_bank, ZoneType.donor_center):
        if low_inventory or cap_remaining > 10:
            need_counts: dict = {}
            for z in obs.zones:
                if z.zone_type == ZoneType.hospital:
                    for nt, nq in z.needs.items():
                        if nq > 0:
                            for dt in COMPATIBILITY.get(nt, []):
                                need_counts[dt] = need_counts.get(dt, 0) + nq
            best_bt, best_val = None, -1.0
            for bt in BLOOD_TYPES:
                stock = current_zone.stock.get(bt, 0)
                if stock <= 0:
                    continue
                val = need_counts.get(bt, 0) - inventory.get(bt, 0) * 0.5
                if val > best_val:
                    best_val, best_bt = val, bt
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
    def _can_deliver(z: object) -> bool:
        for nt, nq in z.needs.items():  # type: ignore[attr-defined]
            if nq <= 0:
                continue
            for dt in COMPATIBILITY.get(nt, []):
                if inventory.get(dt, 0) > 0:
                    return True
        return False

    def _score(z: object) -> float:
        dist = _bfs_dist(ax, ay, z.x, z.y, blocked)  # type: ignore[attr-defined]
        need = sum(z.needs.values())  # type: ignore[attr-defined]
        urgency_val = {"critical": 1000, "high": 400, "moderate": 100,
                       "low": 30, "stable": 5}.get(z.urgency.value, 5)  # type: ignore[attr-defined]
        dying = (z.urgency.value == "critical" and z.steps_unserved >= 3) or \
                (z.urgency.value == "high" and z.steps_unserved >= 5)  # type: ignore[attr-defined]
        will_die = (z.urgency.value == "critical" and z.steps_unserved + dist >= 3) or \
                   (z.urgency.value == "high" and z.steps_unserved + dist >= 5)  # type: ignore[attr-defined]
        if dying:
            urgency_val *= 5
        elif will_die:
            urgency_val *= 2
        return urgency_val * need / (dist + 1)

    hospitals = [z for z in obs.zones
                 if z.zone_type == ZoneType.hospital and sum(z.needs.values()) > 0]
    deliverable = [z for z in hospitals if _can_deliver(z)]

    if deliverable and not low_inventory:
        target = max(deliverable, key=_score)
    else:
        sources = [z for z in obs.zones
                   if z.zone_type in (ZoneType.blood_bank, ZoneType.donor_center)
                   and sum(z.stock.values()) > 0]
        target = min(sources, key=lambda z: _bfs_dist(ax, ay, z.x, z.y, blocked)) \
            if sources else None

    if target is None:
        return DeliveryAction(action_type=ActionType.wait)
    direction = _bfs_direction(ax, ay, target.x, target.y, blocked)
    if direction:
        return DeliveryAction(action_type=ActionType.move, direction=direction)
    return DeliveryAction(action_type=ActionType.wait)

# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

def _parse_llm_action(text: str) -> Optional[DeliveryAction]:
    """Parse JSON action from LLM response text."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return DeliveryAction(**data)
    except Exception:
        return None


def _llm_action(obs: BloodObservation) -> tuple[DeliveryAction, dict]:
    """Call LLM for next action. Falls back to greedy on failure."""
    prompt = _obs_to_prompt(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=150,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        action = _parse_llm_action(raw)
        if action is not None:
            return action, json.loads(action.model_dump_json())
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}", file=sys.stderr)

    fallback = _fallback_action(obs)
    return fallback, json.loads(fallback.model_dump_json())

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
        action, action_dict = _llm_action(obs)
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
        "score": max(0.0001, min(0.9999, round(score, 4))),
        "lives_saved_pct": max(0.0001, min(0.9999, round(lives_pct / 100.0, 4))),
        "steps_used": step_count,
        "mission_success": success,
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set – LLM requests may fail", file=sys.stderr)

    results = []
    for task_id, scenario_name in TASKS:
        result = await run_task(task_id, scenario_name, RNG_SEED)
        results.append(result)
        print(
            f"[INFO] {task_id}: score={result['score']}  lives={result['lives_saved_pct']}%  "
            f"success={result['mission_success']}",
            file=sys.stderr,
        )

    avg_score = max(0.0001, min(0.9999, round(sum(r["score"] for r in results) / len(results), 4)))
    print(f"\n[SUMMARY] avg_score={avg_score}", file=sys.stderr)
    print(json.dumps({"summary": results, "avg_score": avg_score}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
