"""
Server entry point for the Blood Bank Supply Agent OpenEnv environment.
"""
import sys
import os

# Ensure the repo root is on the path so environment.py can be imported
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Optional
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

from environment import (
    BloodBankEnvironment,
    BloodObservation,
    DeliveryAction,
)

app = FastAPI(title="Blood Bank Supply Agent", version="1.0.0")

_env: BloodBankEnvironment = BloodBankEnvironment("city_shortage", 42)


class ResetRequest:
    def __init__(self, scenario: str = "city_shortage", seed: int = 42):
        self.scenario = scenario
        self.seed = seed


from pydantic import BaseModel


class ResetRequestModel(BaseModel):
    scenario: Optional[str] = "city_shortage"
    seed: Optional[int] = 42


class StepRequestModel(BaseModel):
    action: DeliveryAction


@app.get("/health")
async def health():
    return {"status": "ok", "project": "Blood Bank Supply Agent"}


@app.post("/reset")
async def reset(req: Optional[ResetRequestModel] = Body(default=None)):
    global _env
    scenario = (req.scenario if req and req.scenario else None) or "city_shortage"
    seed = (req.seed if req and req.seed is not None else None) or 42
    _env = BloodBankEnvironment(scenario, seed)
    obs = await _env.reset()
    return obs


@app.post("/step")
async def step(req: StepRequestModel):
    obs, reward, done, info = await _env.step(req.action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
async def get_state():
    return _env.state


@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {"id": "easy",   "scenario": "city_shortage",       "difficulty": "easy",   "max_steps": 70},
            {"id": "medium", "scenario": "rare_type_emergency",  "difficulty": "medium", "max_steps": 55},
            {"id": "hard",   "scenario": "disaster_response",    "difficulty": "hard",   "max_steps": 65},
        ]
    }


@app.get("/grader")
async def grader():
    """Run all 3 tasks and return per-task scores strictly in (0.001, 0.999)."""
    import sys
    import os
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from environment import SCENARIOS, BloodBankEnvironment as _BBEnv
    # Import greedy action from root app
    sys.path.insert(0, _ROOT)
    from app import _greedy_action

    def _safe(v: float) -> float:
        return max(0.001, min(0.999, float(v)))

    def _compute(lives_pct, steps_used, max_steps, capacity, cap_remaining):
        lives_frac = _safe(lives_pct / 100.0)
        util = _safe((capacity - cap_remaining) / capacity) if capacity > 0 else 0.001
        speed = _safe(1.0 - steps_used / max_steps) if max_steps > 0 else 0.001
        score = _safe(0.70 * lives_frac + 0.15 * util + 0.15 * speed)
        return {"score": score, "lives_saved_pct": lives_frac,
                "utilization": util, "speed": speed}

    task_scores = {}
    tasks_list = []
    for task_id, scenario_name, max_s in [
        ("easy",   "city_shortage",      70),
        ("medium", "rare_type_emergency", 55),
        ("hard",   "disaster_response",   65),
    ]:
        try:
            env = _BBEnv(scenario_name, rng_seed=99)
            obs = await env.reset()
            done = False
            steps = 0
            while not done:
                action = _greedy_action(obs)
                obs, _r, done, _i = await env.step(action)
                steps += 1
            st = env.state
            lives_pct = float(st.get("lives_saved_pct", 0.0))
            capacity = SCENARIOS.get(scenario_name, {}).get("capacity", 100)
            cap_remaining = int(st.get("capacity_remaining", capacity))
            metrics = _compute(lives_pct, steps, max_s, capacity, cap_remaining)
        except Exception:
            metrics = {"score": 0.001, "lives_saved_pct": 0.001,
                       "utilization": 0.001, "speed": 0.001}
        task_scores[task_id] = metrics["score"]
        tasks_list.append({"task_id": task_id, "scenario": scenario_name, **metrics})

    avg = _safe(sum(task_scores.values()) / len(task_scores))
    return {"score": avg, "task_scores": task_scores, "tasks": tasks_list,
            "breakdown": {"weights": {"lives_saved": 0.70, "utilization": 0.15, "speed": 0.15}}}


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
