---
title: Blood Bank Supply Agent
emoji: 🩸
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# Blood Bank Supply Agent

An OpenEnv-compatible reinforcement learning environment that simulates real-world blood supply logistics. An AI agent navigates a 10×10 grid to collect blood from blood banks and donor centres, then deliver it to hospitals before patients die. The mission succeeds when ≥85% of patients are saved.

---

## Table of Contents

- [Overview](#overview)
- [Environment Design](#environment-design)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Reward Function](#reward-function)
  - [Task Scenarios](#task-scenarios)
- [Setup](#setup)
- [Running the Agent](#running-the-agent)
- [API Reference](#api-reference)
- [Deployment](#deployment)

---

## Overview

| Property | Value |
|----------|-------|
| Grid size | 10 × 10 |
| Zone types | hospital, blood_bank, donor_center, blocked, empty |
| Blood types | O+, O−, A+, A−, B+, B−, AB+, AB− |
| Difficulty tiers | Easy · Medium · Hard |
| Success threshold | ≥ 85% patients saved |
| API standard | OpenEnv (`reset` / `step` / `state`) |

---

## Environment Design

### Action Space

Each step the agent submits one `DeliveryAction` with the following structure:

```json
{
  "action_type": "move | deliver | collect | wait",
  "direction":      "north | south | east | west",  // required for move
  "target_zone_id": "Z_x_y",                        // required for deliver / collect
  "blood_type":     "O+ | O- | A+ | A- | B+ | B- | AB+ | AB-",  // deliver / collect
  "quantity":       10                               // 1–50, deliver / collect
}
```

| action_type | Description |
|-------------|-------------|
| `move`      | Move agent one cell in the given direction (blocked cells are impassable) |
| `deliver`   | Deliver blood from agent inventory to the target hospital |
| `collect`   | Pick up blood from a blood bank or donor centre |
| `wait`      | Agent stays in place for one step |

Blood compatibility follows real-world transfusion rules (e.g. O− is universal donor).

---

### Observation Space

Each call to `step()` or `reset()` returns a `BloodObservation` object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `scenario_name` | str | Active scenario identifier |
| `step_number` | int | Current step (0 = just reset) |
| `max_steps` | int | Episode length limit |
| `zones` | List[ZoneInfo] | All grid zones with position, type, urgency, needs/stock |
| `critical_hospitals` | List[str] | Zone IDs of hospitals at CRITICAL urgency |
| `blood_banks` | List[str] | Zone IDs of blood banks |
| `donor_centers` | List[str] | Zone IDs of donor centres |
| `agent` | AgentStatus | Current position, inventory, capacity |
| `total_patients` | int | Cumulative patients across all hospitals |
| `patients_saved` | int | Patients who received blood in time |
| `patients_lost` | int | Patients who died waiting |
| `lives_saved_pct` | float | `patients_saved / total_patients × 100` |
| `last_action_result` | str | Human-readable result of the previous action |
| `last_reward` | float | Reward from the previous step |
| `is_complete` | bool | True when episode has ended |
| `mission_success` | bool | True if `lives_saved_pct ≥ 85` |

**ZoneInfo fields (hospitals):** `zone_id`, `x`, `y`, `zone_type`, `name`, `urgency`, `patients_waiting`, `patients_saved`, `patients_lost`, `needs` (blood_type → units needed)

**ZoneInfo fields (blood sources):** same as above plus `stock` (blood_type → units available)

**AgentStatus fields:** `current_zone_id`, `x`, `y`, `inventory` (blood_type → units), `capacity_remaining`, `total_units`

---

### Reward Function

| Event | Reward |
|-------|--------|
| Successful move | −0.05 |
| Invalid move (blocked / out of bounds) | −0.20 |
| Deliver to critical hospital | +1.20 |
| Deliver to high urgency hospital | +0.80 |
| Deliver to moderate urgency hospital | +0.50 |
| Deliver to low urgency hospital | +0.20 |
| Deliver to stable hospital | +0.05 |
| Collect blood | +0.10 |
| Wait action | −0.10 |
| Patient waiting time penalty (per step) | variable (negative) |
| Terminal – coverage bonus | `min(1.0, lives_pct/100) × 6.0` |
| Terminal – speed bonus | `max(0.0, 1 − step/max_steps) × 2.0` |
| Terminal – mission success | +10.0 |

**Composite score (0.0–1.0):**
```
score = 0.70 × (lives_saved_pct / 100)
      + 0.15 × utilization
      + 0.15 × speed_bonus
```

---

### Task Scenarios

| ID | Difficulty | Display Name | Max Steps | Hospitals | Blocked Zones | Start |
|----|-----------|--------------|-----------|-----------|---------------|-------|
| `city_shortage` | Easy | Mumbai City Blood Shortage | 70 | 4 | 0 | (5, 5) |
| `rare_type_emergency` | Medium | Delhi Rare Blood Emergency | 55 | 7 | 6 | (0, 9) |
| `disaster_response` | Hard | Chennai Disaster Response | 65 | 8 | 12 | (0, 0) |

- **Easy** – Straightforward logistics, no obstacles, moderate blood demands.
- **Medium** – Rare blood types (O−, AB−, B−) required; 6 blocked zones force pathfinding.
- **Hard** – 3 critical hospitals, 12 blocked zones, minimal starting inventory, tight capacity.

---

## Setup

### Requirements

- Python 3.11+
- See `requirements.txt`

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0
python-dotenv>=1.0.0
openai>=1.0.0
httpx>=0.27.0
openenv-core>=0.2.0
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace bearer token for model API access |
| `API_BASE_URL` | Yes | OpenAI-compatible endpoint (default: HuggingFace router) |
| `MODEL_NAME` | Yes | Model identifier (default: `Qwen/Qwen2.5-72B-Instruct`) |
| `RNG_SEED` | No | RNG seed for reproducibility (default: `42`) |

### Install

```bash
pip install -r requirements.txt
```

---

## Running the Agent

### LLM inference agent (all 3 tasks)

```bash
python inference.py
```

Runs all three scenarios using the configured LLM. Falls back to the greedy heuristic if the LLM call fails.

### Greedy baseline (reproducible scores)

```bash
python baseline.py
```

Runs all three scenarios with a deterministic greedy heuristic. No API keys required.

### Interactive web server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860` for the interactive dashboard with manual controls and live grid visualisation.

---

### Log Format

Both `inference.py` and `baseline.py` emit structured logs to stdout:

```
[START] {"task_id": "easy", "scenario": "city_shortage", "seed": 42}
[STEP]  {"step": 1, "action": {...}, "reward": -0.05, "lives_saved_pct": 0.0, ...}
[STEP]  {"step": 2, ...}
...
[END]   {"task_id": "easy", "score": 0.8234, "lives_saved_pct": 91.5, "steps": 58, "mission_success": true}
```

---

## API Reference

The FastAPI server exposes the full OpenEnv interface:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check – returns `{"status": "ok"}` |
| `/reset` | POST | Initialise a new episode. Body: `{"scenario": "city_shortage", "seed": 42}` |
| `/step` | POST | Execute one action. Body: `{"action": {DeliveryAction}}` |
| `/state` | GET | Full environment state as JSON |
| `/tasks` | GET | List all available scenarios with descriptions |
| `/grader` | GET | Compute composite score for the current episode |
| `/baseline` | GET | Run greedy agent on all 3 tasks and return scores |

Auto-generated interactive docs available at `http://localhost:7860/docs`.

---

## Deployment

### Docker (local)

```bash
docker build -t blood-bank-agent .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  blood-bank-agent
```

### HuggingFace Spaces

1. Create a new Space (Docker SDK) at huggingface.co/spaces
2. Push this repository to the Space
3. Add `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` as Space secrets
4. The Space will expose the environment on port 7860

Infrastructure requirements: 2 vCPU · 8 GB RAM · inference runtime < 20 minutes
