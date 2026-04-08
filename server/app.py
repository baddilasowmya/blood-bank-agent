"""
Server entry point for the Blood Bank Supply Agent OpenEnv environment.
"""
import sys
import os
import json as _json

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
    SCENARIOS,
)

app = FastAPI(title="Blood Bank Supply Agent", version="1.0.0")

_env: BloodBankEnvironment = BloodBankEnvironment("city_shortage", 42)
_last_obs: Optional[BloodObservation] = None


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


# ---------------------------------------------------------------------------
# Float-clamping helpers — all float fields must be strictly (0.0001, 0.9999)
# ---------------------------------------------------------------------------

def _sf(v: float) -> float:
    """Clamp a float to strictly (0.0001, 0.9999)."""
    return max(0.0001, min(0.9999, round(float(v), 4)))


def _clamp_all(obj):
    """Recursively clamp every float in a response object to (0.0001, 0.9999)."""
    if isinstance(obj, float):
        return _sf(obj)
    if isinstance(obj, dict):
        return {k: _clamp_all(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clamp_all(v) for v in obj]
    return obj


def _normalize_obs(obs: BloodObservation) -> dict:
    """Convert observation to dict with all floats in (0.0001, 0.9999).
    lives_saved_pct is a percentage (0-100) so it is divided by 100 first.
    """
    d = _json.loads(obs.model_dump_json())
    if "lives_saved_pct" in d:
        d["lives_saved_pct"] = _sf(d["lives_saved_pct"] / 100.0)
    if "last_reward" in d:
        d["last_reward"] = _sf(d["last_reward"])
    return d


@app.get("/health")
async def health():
    return {"status": "ok", "project": "Blood Bank Supply Agent"}


@app.post("/reset")
async def reset(req: Optional[ResetRequestModel] = Body(default=None)):
    global _env, _last_obs
    scenario = (req.scenario if req and req.scenario else None) or "city_shortage"
    seed = (req.seed if req and req.seed is not None else None) or 42
    _env = BloodBankEnvironment(scenario, seed)
    _last_obs = await _env.reset()
    return _normalize_obs(_last_obs)


@app.post("/step")
async def step(req: StepRequestModel):
    global _last_obs
    obs, reward, done, info = await _env.step(req.action)
    _last_obs = obs
    return {
        "observation": _normalize_obs(obs),
        "reward": _sf(reward),
        "done": done,
        "info": _clamp_all(info),
    }


@app.get("/state")
async def get_state():
    st = dict(_env.state)
    if "lives_saved_pct" in st:
        st["lives_saved_pct"] = _sf(st["lives_saved_pct"] / 100.0)
    return _clamp_all(st)


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
        return {"score": 0.0001, "lives_saved_pct": 0.0001}

    lives_pct = max(0.0001, st.get("lives_saved_pct", 0.0))
    step = st.get("step", 0)
    max_steps = st.get("max_steps", 70)
    scenario = st.get("scenario", "city_shortage")
    capacity = SCENARIOS.get(scenario, {}).get("capacity", 100)
    cap_remaining = st.get("capacity_remaining", capacity)

    utilization = _sf(max(0.0001, 1.0 - cap_remaining / capacity) if capacity > 0 else 0.0001)
    speed = _sf(max(0.0001, 1.0 - step / max_steps) if max_steps > 0 else 0.0001)
    lives_frac = _sf(lives_pct / 100.0)

    score = _sf(0.7 * lives_frac + 0.15 * utilization + 0.15 * speed)

    return {"score": score, "lives_saved_pct": lives_frac}


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
