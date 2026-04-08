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
    API_BASE_URL  – OpenAI-compatible endpoint (e.g. HuggingFace router)
    MODEL_NAME    – Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      – HuggingFace bearer token
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import re
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
)

load_dotenv()



API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str     = os.environ.get("HF_TOKEN",     "")
RNG_SEED: int     = int(os.environ.get("RNG_SEED",  "42"))

TASKS = [
    ("easy",   "city_shortage"),
    ("medium", "rare_type_emergency"),
    ("hard",   "disaster_response"),
]



client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)



def log_start(task_id: str, scenario: str, seed: int) -> None:
    print(json.dumps({"event": "START", "task_id": task_id,
                      "scenario": scenario, "seed": seed}),
          flush=True)
    # Also emit the [START] prefixed form required by the spec
    sys.stdout.write(f"[START] {{\"task_id\": \"{task_id}\", \"scenario\": \"{scenario}\", \"seed\": {seed}}}\n")
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



def _obs_to_prompt(obs: BloodObservation) -> str:
    agent = obs.agent
    lines = [
        f"=== Blood Bank Supply Agent – Step {obs.step_number}/{obs.max_steps} ===",
        f"Scenario: {obs.scenario_name}",
        f"Agent position: ({agent.x}, {agent.y})  zone={agent.current_zone_id}",
        f"Inventory ({agent.total_units}/{agent.total_units + agent.capacity_remaining}): "
        + ", ".join(f"{bt}:{agent.inventory.get(bt, 0)}" for bt in BLOOD_TYPES if agent.inventory.get(bt, 0) > 0),
        f"Patients – total:{obs.total_patients}  saved:{obs.patients_saved}  lost:{obs.patients_lost}  ({obs.lives_saved_pct:.1f}% saved)",
        "",
        "HOSPITALS:",
    ]

    for z in obs.zones:
        if z.zone_type == ZoneType.hospital and sum(z.needs.values()) > 0:
            needs_str = ", ".join(f"{bt}:{qty}" for bt, qty in z.needs.items() if qty > 0)
            lines.append(
                f"  [{z.urgency.value.upper()}] {z.name} ({z.zone_id}) at ({z.x},{z.y}) "
                f"needs: {needs_str}  waiting:{z.patients_waiting}"
            )

    lines += ["", "BLOOD SOURCES:"]
    for z in obs.zones:
        if z.zone_type in (ZoneType.blood_bank, ZoneType.donor_center):
            stock_str = ", ".join(f"{bt}:{qty}" for bt, qty in z.stock.items() if qty > 0)
            lines.append(
                f"  {z.name} ({z.zone_id}) at ({z.x},{z.y})  stock: {stock_str}"
            )

    lines += [
        "",
        "Last action result: " + obs.last_action_result,
        "",
        "INSTRUCTIONS:",
        "Choose ONE action. Reply with ONLY a valid JSON object (no markdown, no explanation).",
        "Valid action_types: move | deliver | collect | wait",
        "  move:    {\"action_type\": \"move\", \"direction\": \"north|south|east|west\"}",
        "  deliver: {\"action_type\": \"deliver\", \"target_zone_id\": \"Z_x_y\", \"blood_type\": \"O+\", \"quantity\": 10}",
        "  collect: {\"action_type\": \"collect\", \"target_zone_id\": \"Z_x_y\", \"blood_type\": \"O+\", \"quantity\": 20}",
        "  wait:    {\"action_type\": \"wait\"}",
        "",
        "STRATEGY: Prioritise CRITICAL hospitals. Deliver before collecting. Move toward nearest unserved hospital.",
        "If agent is at a hospital and has compatible blood → deliver immediately.",
        "If inventory < 30 units → go collect from nearest blood bank / donor center.",
    ]

    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are an expert logistics AI managing blood bank supply chains. "
    "Your goal is to save ≥85% of patients by delivering compatible blood to hospitals before patients die. "
    "Always reply with a single JSON action object and nothing else."
)



def _fallback_action(obs: BloodObservation) -> DeliveryAction:
    """Simple greedy fallback used when LLM call fails."""
    agent = obs.agent
    ax, ay = agent.x, agent.y
    inventory = agent.inventory

    zone_map = {z.zone_id: z for z in obs.zones}
    current_zone = zone_map.get(agent.current_zone_id)

    # Deliver if at hospital
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

    # Collect if at source and low
    cap_remaining = agent.capacity_remaining
    inv_total = agent.total_units
    capacity = inv_total + cap_remaining
    low_inventory = inv_total < capacity * 0.3

    if current_zone and current_zone.zone_type in (ZoneType.blood_bank, ZoneType.donor_center):
        if low_inventory or cap_remaining > 20:
            for bt in BLOOD_TYPES:
                if current_zone.stock.get(bt, 0) > 0 and cap_remaining > 0:
                    qty = min(cap_remaining, current_zone.stock[bt], 30)
                    return DeliveryAction(
                        action_type=ActionType.collect,
                        target_zone_id=agent.current_zone_id,
                        blood_type=bt,
                        quantity=qty,
                    )

    # Navigate
    blocked = {(z.x, z.y) for z in obs.zones if z.zone_type == ZoneType.blocked}

    if low_inventory:
        candidates = [z for z in obs.zones
                      if z.zone_type in (ZoneType.blood_bank, ZoneType.donor_center)
                      and sum(z.stock.values()) > 0]
    else:
        candidates = sorted(
            [z for z in obs.zones
             if z.zone_type == ZoneType.hospital and sum(z.needs.values()) > 0],
            key=lambda z: (
                {"critical": 0, "high": 1, "moderate": 2, "low": 3, "stable": 4}.get(
                    z.urgency.value, 4),
                abs(z.x - ax) + abs(z.y - ay),
            )
        )

    if not candidates:
        return DeliveryAction(action_type=ActionType.wait)

    target = min(candidates, key=lambda z: abs(z.x - ax) + abs(z.y - ay))

    # Simple BFS direction
    from collections import deque
    queue = deque([(ax, ay, [])])
    visited = {(ax, ay)}
    dir_map = {
        Direction.north: (0, -1), Direction.south: (0, 1),
        Direction.west: (-1, 0),  Direction.east:  (1, 0),
    }
    found_dir: Optional[Direction] = None
    while queue and found_dir is None:
        cx, cy, path = queue.popleft()
        for d, (dx, dy) in dir_map.items():
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < 10 and 0 <= ny < 10):
                continue
            if (nx, ny) in visited or (nx, ny) in blocked:
                continue
            new_path = path + [d]
            if nx == target.x and ny == target.y:
                found_dir = new_path[0]
                break
            visited.add((nx, ny))
            queue.append((nx, ny, new_path))

    if found_dir:
        return DeliveryAction(action_type=ActionType.move, direction=found_dir)

    # Greedy fallback direction
    if target.x > ax:
        return DeliveryAction(action_type=ActionType.move, direction=Direction.east)
    if target.x < ax:
        return DeliveryAction(action_type=ActionType.move, direction=Direction.west)
    if target.y > ay:
        return DeliveryAction(action_type=ActionType.move, direction=Direction.south)
    return DeliveryAction(action_type=ActionType.move, direction=Direction.north)


def _parse_llm_action(text: str) -> Optional[DeliveryAction]:
    """Parse JSON action from LLM response text."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        return DeliveryAction(**data)
    except Exception:
        return None


def _llm_action(obs: BloodObservation) -> tuple[DeliveryAction, dict]:
    """Call LLM for next action. Returns (action, action_dict)."""
    prompt = _obs_to_prompt(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=150,
            temperature=0.2,
        )
        raw = response.choices[0].message.content or ""
        action = _parse_llm_action(raw)
        if action is not None:
            return action, json.loads(action.model_dump_json())
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}", file=sys.stderr)

    fallback = _fallback_action(obs)
    return fallback, json.loads(fallback.model_dump_json())



def _compute_score(lives_pct: float, steps_used: int, max_steps: int, capacity: int,
                   cap_remaining: int) -> float:
    utilization = max(0.0, 1.0 - cap_remaining / capacity) if capacity > 0 else 0.0
    speed = max(0.0, 1.0 - steps_used / max_steps) if max_steps > 0 else 0.0
    score = 0.7 * (lives_pct / 100.0) + 0.15 * utilization + 0.15 * speed
    # Clamp strictly within (0, 1) as required by the evaluator
    return round(max(0.01, min(0.99, score)), 4)



async def run_task(task_id: str, scenario_name: str, seed: int) -> dict:
    log_start(task_id, scenario_name, seed)

    env = BloodBankEnvironment(scenario_name, rng_seed=seed)
    obs = await env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        action, action_dict = _llm_action(obs)
        obs, reward, done, _ = await env.step(action)
        step_count += 1
        total_reward += reward
        log_step(step_count, action_dict, reward, obs)

    st = env.state
    from environment import SCENARIOS
    capacity = SCENARIOS.get(scenario_name, {}).get("capacity", 100)
    cap_remaining = st.get("capacity_remaining", capacity)
    lives_pct = st.get("lives_saved_pct", 0.0)
    score = _compute_score(lives_pct, step_count, obs.max_steps, capacity, cap_remaining)

    log_end(task_id, score, lives_pct, step_count, bool(st.get("mission_success", False)))

    return {
        "task_id": task_id,
        "scenario": scenario_name,
        "score": score,
    }



async def main() -> None:
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set – requests may be rejected", file=sys.stderr)

    results = []
    for task_id, scenario_name in TASKS:
        result = await run_task(task_id, scenario_name, RNG_SEED)
        results.append(result)
        print(f"[INFO] {task_id}: score={result['score']}  lives={result['lives_saved_pct']}%  "
              f"success={result['mission_success']}", file=sys.stderr)

    # Final summary
    avg_score = round(sum(r["score"] for r in results) / len(results), 4)
    print(f"\n[SUMMARY] avg_score={avg_score}", file=sys.stderr)
    print(json.dumps({"summary": results, "avg_score": avg_score}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
