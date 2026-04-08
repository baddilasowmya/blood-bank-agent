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
    st = _env.state
    if st.get("status") == "not_initialized":
        return {"score": 0.01}
    lives_pct = st.get("lives_saved_pct", 0.01)
    step = st.get("step", 0)
    max_steps = st.get("max_steps", 70)
    from environment import SCENARIOS
    scenario = st.get("scenario", "city_shortage")
    capacity = SCENARIOS.get(scenario, {}).get("capacity", 100)
    cap_remaining = st.get("capacity_remaining", capacity)
    utilization = max(0.0, 1.0 - cap_remaining / capacity) if capacity > 0 else 0.0
    speed = max(0.0, 1.0 - step / max_steps) if max_steps > 0 else 0.0
    raw_score = 0.7 * (lives_pct / 100.0) + 0.15 * utilization + 0.15 * speed
    # Clamp strictly within (0, 1) as required by the evaluator
    score = round(max(0.01, min(0.99, raw_score)), 4)
    return {"score": score, "lives_saved_pct": lives_pct}


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
