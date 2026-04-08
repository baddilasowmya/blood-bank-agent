"""
Blood Bank Supply Agent - RL Environment
Optimizes blood supply logistics across hospitals to prevent patient deaths.
"""
from __future__ import annotations

import random
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field




class ActionType(str, Enum):
    deliver = "deliver"
    collect = "collect"
    move = "move"
    wait = "wait"


class Direction(str, Enum):
    north = "north"
    south = "south"
    east = "east"
    west = "west"


class UrgencyLevel(str, Enum):
    critical = "critical"
    high = "high"
    moderate = "moderate"
    low = "low"
    stable = "stable"


class ZoneType(str, Enum):
    hospital = "hospital"
    blood_bank = "blood_bank"
    donor_center = "donor_center"
    empty = "empty"
    blocked = "blocked"


BLOOD_TYPES: List[str] = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]


COMPATIBILITY: Dict[str, List[str]] = {
    "O+":  ["O+", "O-"],
    "O-":  ["O-"],
    "A+":  ["A+", "A-", "O+", "O-"],
    "A-":  ["A-", "O-"],
    "B+":  ["B+", "B-", "O+", "O-"],
    "B-":  ["B-", "O-"],
    "AB+": ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"],
    "AB-": ["AB-", "A-", "B-", "O-"],
}

URGENCY_REWARDS: Dict[str, float] = {
    "critical": 1.2,
    "high":     0.8,
    "moderate": 0.5,
    "low":      0.2,
    "stable":   0.05,
}




class ZoneInfo(BaseModel):
    zone_id: str
    x: int
    y: int
    zone_type: ZoneType
    name: str
    # hospitals: blood type -> units needed
    needs: Dict[str, int] = Field(default_factory=dict)
    units_delivered: Dict[str, int] = Field(default_factory=dict)
    urgency: UrgencyLevel = UrgencyLevel.stable
    patients_waiting: int = 0
    patients_saved: int = 0
    patients_lost: int = 0
    steps_unserved: int = 0
    # blood banks / donor centers: blood type -> units in stock
    stock: Dict[str, int] = Field(default_factory=dict)
    is_accessible: bool = True


class AgentStatus(BaseModel):
    current_zone_id: str
    x: int
    y: int
    inventory: Dict[str, int] = Field(default_factory=dict)
    capacity_remaining: int
    total_units: int


class DeliveryAction(BaseModel):
    action_type: ActionType
    target_zone_id: Optional[str] = None
    blood_type: Optional[str] = None
    quantity: Optional[int] = Field(default=None, ge=1, le=50)
    direction: Optional[Direction] = None


class BloodObservation(BaseModel):
    scenario_name: str
    step_number: int
    max_steps: int
    zones: List[ZoneInfo]
    critical_hospitals: List[str]
    blood_banks: List[str]
    donor_centers: List[str]
    agent: AgentStatus
    total_patients: int
    patients_saved: int
    patients_lost: int
    lives_saved_pct: float
    last_action_result: str
    last_reward: float
    is_complete: bool
    mission_success: bool



def _mumbai_city_shortage() -> dict:
    """Easy scenario – 70 steps."""
    return {
        "name": "city_shortage",
        "display_name": "Mumbai City Blood Shortage",
        "difficulty": "easy",
        "max_steps": 70,
        "agent_start": (5, 5),
        "initial_inventory": {
            "O+": 20, "A+": 15, "B+": 15, "O-": 10,
            "A-": 5,  "B-": 5,  "AB+": 5, "AB-": 3
        },
        "capacity": 120,
        "zones": [
            # Blood banks
            {
                "zone_id": "Z_4_4", "x": 4, "y": 4,
                "zone_type": "blood_bank", "name": "Central Blood Bank",
                "stock": {"O+": 100, "A+": 80, "B+": 60, "O-": 30,
                          "A-": 20, "B-": 20, "AB+": 15, "AB-": 10},
            },
            {
                "zone_id": "Z_6_4", "x": 6, "y": 4,
                "zone_type": "blood_bank", "name": "West Blood Bank",
                "stock": {"O+": 80, "A+": 60, "B+": 50, "O-": 20,
                          "A-": 15, "B-": 15, "AB+": 10, "AB-": 8},
            },
            # Donor center
            {
                "zone_id": "Z_5_0", "x": 5, "y": 0,
                "zone_type": "donor_center", "name": "City Donor Center",
                "stock": {"O+": 50, "A+": 40, "B+": 30, "O-": 15,
                          "A-": 10, "B-": 10, "AB+": 8, "AB-": 5},
            },
            # Hospitals
            {
                "zone_id": "Z_2_2", "x": 2, "y": 2,
                "zone_type": "hospital", "name": "Mumbai General",
                "urgency": "high", "patients_waiting": 80,
                "needs": {"O+": 20, "A+": 15, "B+": 10},
            },
            {
                "zone_id": "Z_8_2", "x": 8, "y": 2,
                "zone_type": "hospital", "name": "Northern Clinic",
                "urgency": "moderate", "patients_waiting": 60,
                "needs": {"A+": 15, "O+": 12, "AB+": 6},
            },
            {
                "zone_id": "Z_2_7", "x": 2, "y": 7,
                "zone_type": "hospital", "name": "Western Hospital",
                "urgency": "moderate", "patients_waiting": 50,
                "needs": {"B+": 12, "O+": 10, "A-": 8},
            },
            {
                "zone_id": "Z_8_7", "x": 8, "y": 7,
                "zone_type": "hospital", "name": "Southern Medical",
                "urgency": "low", "patients_waiting": 40,
                "needs": {"O+": 15, "B+": 10, "O-": 6},
            },
        ],
        "blocked": [],
    }


def _delhi_rare_blood_emergency() -> dict:
    """Medium scenario – 55 steps."""
    return {
        "name": "rare_type_emergency",
        "display_name": "Delhi Rare Blood Emergency",
        "difficulty": "medium",
        "max_steps": 55,
        "agent_start": (0, 9),
        "initial_inventory": {
            "O+": 15, "A+": 12, "B+": 12, "O-": 8,
            "A-": 4,  "B-": 4,  "AB+": 3, "AB-": 4
        },
        "capacity": 100,
        "zones": [
            # Blood banks
            {
                "zone_id": "Z_0_9", "x": 0, "y": 9,
                "zone_type": "blood_bank", "name": "South-West Blood Bank",
                "stock": {"O+": 60, "A+": 40, "B+": 30, "O-": 10,
                          "A-": 8, "B-": 8, "AB+": 5, "AB-": 5},
            },
            {
                "zone_id": "Z_9_9", "x": 9, "y": 9,
                "zone_type": "blood_bank", "name": "South-East Blood Bank",
                "stock": {"O+": 50, "B+": 35, "A+": 30, "O-": 8,
                          "AB-": 6, "A-": 6, "B-": 6, "AB+": 4},
            },
            # Donor centers
            {
                "zone_id": "Z_2_0", "x": 2, "y": 0,
                "zone_type": "donor_center", "name": "North-West Donor Center",
                "stock": {"O+": 30, "A+": 20, "O-": 8, "B+": 20,
                          "A-": 5, "B-": 5, "AB+": 3, "AB-": 3},
            },
            {
                "zone_id": "Z_7_0", "x": 7, "y": 0,
                "zone_type": "donor_center", "name": "North-East Donor Center",
                "stock": {"O+": 25, "B+": 18, "AB-": 8, "O-": 6,
                          "A+": 15, "A-": 4, "B-": 4, "AB+": 3},
            },
            # Hospitals
            {
                "zone_id": "Z_2_2", "x": 2, "y": 2,
                "zone_type": "hospital", "name": "AIIMS Delhi",
                "urgency": "critical", "patients_waiting": 120,
                "needs": {"O-": 20, "AB-": 15, "A-": 12},
            },
            {
                "zone_id": "Z_7_2", "x": 7, "y": 2,
                "zone_type": "hospital", "name": "Ram Manohar Hospital",
                "urgency": "critical", "patients_waiting": 100,
                "needs": {"AB-": 18, "O-": 15, "B-": 10},
            },
            {
                "zone_id": "Z_5_4", "x": 5, "y": 4,
                "zone_type": "hospital", "name": "Central Govt Hospital",
                "urgency": "high", "patients_waiting": 90,
                "needs": {"O+": 25, "A+": 18, "B+": 12},
            },
            {
                "zone_id": "Z_1_6", "x": 1, "y": 6,
                "zone_type": "hospital", "name": "West Delhi Clinic",
                "urgency": "high", "patients_waiting": 70,
                "needs": {"A+": 15, "O+": 12, "A-": 8},
            },
            {
                "zone_id": "Z_8_5", "x": 8, "y": 5,
                "zone_type": "hospital", "name": "East Medical Centre",
                "urgency": "moderate", "patients_waiting": 60,
                "needs": {"B+": 14, "O+": 10, "AB+": 6},
            },
            {
                "zone_id": "Z_4_7", "x": 4, "y": 7,
                "zone_type": "hospital", "name": "South Delhi Hospital",
                "urgency": "moderate", "patients_waiting": 50,
                "needs": {"O+": 12, "B+": 10, "A+": 8},
            },
            {
                "zone_id": "Z_8_8", "x": 8, "y": 8,
                "zone_type": "hospital", "name": "Border Hospital",
                "urgency": "low", "patients_waiting": 40,
                "needs": {"O+": 10, "A+": 8, "B-": 5},
            },
        ],
        "blocked": [(3, 3), (6, 3), (4, 5), (7, 5), (2, 6), (8, 6)],
    }


def _chennai_disaster_response() -> dict:
    """Hard scenario – 65 steps."""
    return {
        "name": "disaster_response",
        "display_name": "Chennai Disaster Response",
        "difficulty": "hard",
        "max_steps": 65,
        "agent_start": (0, 0),
        "initial_inventory": {
            "O+": 15, "O-": 12, "A+": 10, "B+": 10,
            "A-": 3,  "B-": 3,  "AB+": 2, "AB-": 2
        },
        "capacity": 80,
        "zones": [
            # Blood banks
            {
                "zone_id": "Z_0_0", "x": 0, "y": 0,
                "zone_type": "blood_bank", "name": "North Blood Bank (Damaged)",
                "stock": {"O+": 80, "O-": 40, "A+": 50, "B+": 40,
                          "A-": 15, "B-": 15, "AB+": 10, "AB-": 8},
            },
            {
                "zone_id": "Z_9_9", "x": 9, "y": 9,
                "zone_type": "blood_bank", "name": "South Blood Bank",
                "stock": {"O+": 60, "O-": 30, "A+": 40, "B+": 30,
                          "A-": 12, "B-": 12, "AB+": 8, "AB-": 6},
            },
            # Donor center
            {
                "zone_id": "Z_5_5", "x": 5, "y": 5,
                "zone_type": "donor_center", "name": "Central Relief Donor",
                "stock": {"O+": 40, "O-": 20, "A+": 25, "B+": 20,
                          "A-": 8, "B-": 8, "AB+": 5, "AB-": 4},
            },
            # Hospitals – 8 total, 3 critical
            {
                "zone_id": "Z_2_0", "x": 2, "y": 0,
                "zone_type": "hospital", "name": "Chennai Trauma Center",
                "urgency": "critical", "patients_waiting": 150,
                "needs": {"O+": 30, "O-": 25, "A+": 20},
            },
            {
                "zone_id": "Z_7_1", "x": 7, "y": 1,
                "zone_type": "hospital", "name": "Rajiv Gandhi Hospital",
                "urgency": "critical", "patients_waiting": 120,
                "needs": {"O-": 22, "O+": 25, "B+": 15},
            },
            {
                "zone_id": "Z_4_4", "x": 4, "y": 4,
                "zone_type": "hospital", "name": "Government General",
                "urgency": "critical", "patients_waiting": 100,
                "needs": {"O+": 20, "A+": 18, "O-": 15},
            },
            {
                "zone_id": "Z_1_4", "x": 1, "y": 4,
                "zone_type": "hospital", "name": "Western Relief Camp",
                "urgency": "high", "patients_waiting": 80,
                "needs": {"O+": 18, "A+": 12, "B+": 10},
            },
            {
                "zone_id": "Z_8_4", "x": 8, "y": 4,
                "zone_type": "hospital", "name": "Coastal Hospital",
                "urgency": "high", "patients_waiting": 70,
                "needs": {"B+": 15, "O+": 12, "A-": 8},
            },
            {
                "zone_id": "Z_3_7", "x": 3, "y": 7,
                "zone_type": "hospital", "name": "Suburban Clinic",
                "urgency": "moderate", "patients_waiting": 60,
                "needs": {"A+": 12, "O+": 10, "B-": 6},
            },
            {
                "zone_id": "Z_7_6", "x": 7, "y": 6,
                "zone_type": "hospital", "name": "Southern Medical",
                "urgency": "moderate", "patients_waiting": 50,
                "needs": {"B+": 10, "O+": 8, "AB+": 5},
            },
            {
                "zone_id": "Z_5_9", "x": 5, "y": 9,
                "zone_type": "hospital", "name": "Port Hospital",
                "urgency": "low", "patients_waiting": 40,
                "needs": {"O+": 10, "A+": 8, "O-": 5},
            },
        ],
        "blocked": [
            (1, 1), (2, 3), (4, 2), (6, 1), (8, 2),
            (3, 5), (7, 4), (5, 6), (1, 7), (8, 7), (3, 8), (6, 8),
        ],
    }


SCENARIOS: Dict[str, dict] = {
    "city_shortage":       _mumbai_city_shortage(),
    "rare_type_emergency": _delhi_rare_blood_emergency(),
    "disaster_response":   _chennai_disaster_response(),
}


def get_scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]




class BloodGrid:
    """10×10 grid that models zone state and agent physics."""

    def __init__(self, scenario: dict, rng_seed: int = 42):
        self._scenario = scenario
        self._rng = random.Random(rng_seed)
        self._zones: Dict[str, ZoneInfo] = {}
        self._grid: Dict[Tuple[int, int], str] = {}  # (x,y) -> zone_id
        self._agent_x: int = 0
        self._agent_y: int = 0
        self._build()

 
    def _build(self) -> None:
        sc = self._scenario
        ax, ay = sc["agent_start"]
        self._agent_x = ax
        self._agent_y = ay

        # Place blocked zones first
        for bx, by in sc.get("blocked", []):
            zid = f"Z_{bx}_{by}"
            z = ZoneInfo(
                zone_id=zid, x=bx, y=by,
                zone_type=ZoneType.blocked,
                name="Blocked",
                is_accessible=False,
            )
            self._zones[zid] = z
            self._grid[(bx, by)] = zid

        # Place defined zones
        for zd in sc["zones"]:
            zid = zd["zone_id"]
            zt = ZoneType(zd["zone_type"])
            urgency = UrgencyLevel(zd.get("urgency", "stable"))
            z = ZoneInfo(
                zone_id=zid,
                x=zd["x"],
                y=zd["y"],
                zone_type=zt,
                name=zd["name"],
                needs=dict(zd.get("needs", {})),
                stock=dict(zd.get("stock", {})),
                urgency=urgency,
                patients_waiting=zd.get("patients_waiting", 0),
                is_accessible=True,
            )
            # initialise units_delivered keys
            for bt in z.needs:
                z.units_delivered[bt] = 0
            self._zones[zid] = z
            self._grid[(zd["x"], zd["y"])] = zid

        # Fill remaining cells with empty zones
        for x in range(10):
            for y in range(10):
                if (x, y) not in self._grid:
                    zid = f"Z_{x}_{y}"
                    z = ZoneInfo(
                        zone_id=zid, x=x, y=y,
                        zone_type=ZoneType.empty,
                        name=f"Empty ({x},{y})",
                        is_accessible=True,
                    )
                    self._zones[zid] = z
                    self._grid[(x, y)] = zid


    def zone(self, x: int, y: int) -> Optional[ZoneInfo]:
        zid = self._grid.get((x, y))
        return self._zones.get(zid) if zid else None

    def zone_by_id(self, zid: str) -> Optional[ZoneInfo]:
        return self._zones.get(zid)

    def all_zones(self) -> List[ZoneInfo]:
        return list(self._zones.values())

    def hospital_zones(self) -> List[ZoneInfo]:
        return [z for z in self._zones.values() if z.zone_type == ZoneType.hospital]

    def blood_source_zones(self) -> List[ZoneInfo]:
        return [z for z in self._zones.values()
                if z.zone_type in (ZoneType.blood_bank, ZoneType.donor_center)]

    def critical_hospital_ids(self) -> List[str]:
        return [z.zone_id for z in self.hospital_zones()
                if z.urgency == UrgencyLevel.critical]

    def agent_zone_id(self) -> str:
        return self._grid[(self._agent_x, self._agent_y)]

    def agent_pos(self) -> Tuple[int, int]:
        return self._agent_x, self._agent_y



    def move_agent(self, direction: Direction) -> Tuple[bool, str]:
        dx, dy = 0, 0
        if direction == Direction.north:
            dy = -1
        elif direction == Direction.south:
            dy = 1
        elif direction == Direction.west:
            dx = -1
        elif direction == Direction.east:
            dx = 1

        nx, ny = self._agent_x + dx, self._agent_y + dy

        if not (0 <= nx < 10 and 0 <= ny < 10):
            return False, f"Cannot move {direction.value}: out of bounds."

        target_zone = self.zone(nx, ny)
        if target_zone and not target_zone.is_accessible:
            return False, f"Cannot move {direction.value}: zone is blocked."

        self._agent_x, self._agent_y = nx, ny
        zid = self._grid[(nx, ny)]
        return True, f"Moved {direction.value} to ({nx},{ny}) [{zid}]."



    def deliver(
        self,
        zone_id: str,
        blood_type: str,
        quantity: int,
        inventory: Dict[str, int],
    ) -> Tuple[float, str, int]:
        """
        Deliver blood_type to hospital zone_id.
        Returns (reward, message, actual_qty_used).
        Uses compatibility rules (O- universal donor, O+ covers positive types).
        """
        zone = self.zone_by_id(zone_id)
        if zone is None:
            return 0.0, f"Zone {zone_id} not found.", 0
        if zone.zone_type != ZoneType.hospital:
            return 0.0, f"{zone_id} is not a hospital.", 0
        ax, ay = self.agent_pos()
        if ax != zone.x or ay != zone.y:
            return 0.0, f"Agent not at {zone_id} (at {ax},{ay}).", 0
        if blood_type not in inventory or inventory[blood_type] <= 0:
            return 0.0, f"No {blood_type} in inventory.", 0

        # Find a hospital need that this blood_type can satisfy
        satisfied_need_type = None
        for need_type, need_qty in zone.needs.items():
            if need_qty > 0 and blood_type in COMPATIBILITY.get(need_type, []):
                satisfied_need_type = need_type
                break

        if satisfied_need_type is None:
            return 0.0, f"Hospital {zone_id} has no need compatible with {blood_type}.", 0

        available = inventory[blood_type]
        needed = zone.needs[satisfied_need_type]
        actual = min(quantity, available, needed)
        if actual <= 0:
            return 0.0, f"Nothing to deliver (need={needed}, have={available}).", 0

        # Update zone
        zone.needs[satisfied_need_type] = max(0, zone.needs[satisfied_need_type] - actual)
        zone.units_delivered[satisfied_need_type] = (
            zone.units_delivered.get(satisfied_need_type, 0) + actual
        )

        # Patients saved proportional to delivery
        patients_helped = max(1, int(actual * zone.patients_waiting /
                                     max(1, sum(zone.needs.values()) + actual)))
        zone.patients_saved += patients_helped
        zone.patients_waiting = max(0, zone.patients_waiting - patients_helped)
        zone.steps_unserved = 0

        # Reward
        urgency_mult = URGENCY_REWARDS.get(zone.urgency.value, 0.2)
        reward = actual * urgency_mult

        self._update_urgency(zone)

        msg = (f"Delivered {actual}u {blood_type} → {zone.name} "
               f"(satisfying {satisfied_need_type}, urgency={zone.urgency.value}). "
               f"+{reward:.2f} reward.")
        return reward, msg, actual

 

    def collect(
        self,
        zone_id: str,
        blood_type: str,
        quantity: int,
    ) -> Tuple[bool, str, int]:
        """
        Collect blood_type from blood bank or donor center.
        Returns (success, message, actual_qty_collected).
        """
        zone = self.zone_by_id(zone_id)
        if zone is None:
            return False, f"Zone {zone_id} not found.", 0
        if zone.zone_type not in (ZoneType.blood_bank, ZoneType.donor_center):
            return False, f"{zone_id} is not a blood source.", 0
        ax, ay = self.agent_pos()
        if ax != zone.x or ay != zone.y:
            return False, f"Agent not at {zone_id} (at {ax},{ay}).", 0
        if blood_type not in BLOOD_TYPES:
            return False, f"Unknown blood type {blood_type}.", 0

        available = zone.stock.get(blood_type, 0)
        if available <= 0:
            return False, f"No {blood_type} available at {zone.name}.", 0

        actual = min(quantity, available)
        zone.stock[blood_type] = available - actual
        return True, f"Collected {actual}u {blood_type} from {zone.name}.", actual


    def _update_urgency(self, zone: ZoneInfo) -> None:
        total_need = sum(zone.needs.values())
        total_delivered = sum(zone.units_delivered.values())
        total_original = total_need + total_delivered
        if total_original == 0:
            zone.urgency = UrgencyLevel.stable
            return
        ratio = total_need / total_original  # fraction still unmet
        if ratio >= 0.75:
            zone.urgency = UrgencyLevel.critical
        elif ratio >= 0.5:
            zone.urgency = UrgencyLevel.high
        elif ratio >= 0.25:
            zone.urgency = UrgencyLevel.moderate
        elif ratio > 0:
            zone.urgency = UrgencyLevel.low
        else:
            zone.urgency = UrgencyLevel.stable


    def advance_time(self) -> Dict[str, float]:
        penalties: Dict[str, float] = {}

        for zone in self.hospital_zones():
            remaining_need = sum(zone.needs.values())
            if remaining_need > 0:
                zone.steps_unserved += 1
                # Escalating penalties for unserved critical/high hospitals
                if zone.urgency == UrgencyLevel.critical:
                    if zone.steps_unserved >= 3:
                        lost = max(1, int(zone.patients_waiting * 0.05))
                        zone.patients_lost += lost
                        zone.patients_waiting = max(0, zone.patients_waiting - lost)
                        penalties[zone.zone_id] = -lost * 0.5
                elif zone.urgency == UrgencyLevel.high:
                    if zone.steps_unserved >= 5:
                        lost = max(1, int(zone.patients_waiting * 0.03))
                        zone.patients_lost += lost
                        zone.patients_waiting = max(0, zone.patients_waiting - lost)
                        penalties[zone.zone_id] = -lost * 0.3
                # Re-evaluate urgency
                self._update_urgency(zone)
            else:
                zone.steps_unserved = 0

            # Random small new patient arrivals
            if self._rng.random() < 0.15:
                new_patients = self._rng.randint(1, 5)
                zone.patients_waiting += new_patients
                # Small random new need
                if zone.needs:
                    bt = self._rng.choice(list(zone.needs.keys()))
                    zone.needs[bt] = zone.needs.get(bt, 0) + self._rng.randint(1, 3)

        # Donor centers regenerate small stock
        for zone in self.blood_source_zones():
            if zone.zone_type == ZoneType.donor_center:
                if self._rng.random() < 0.3:
                    bt = self._rng.choice(BLOOD_TYPES)
                    zone.stock[bt] = zone.stock.get(bt, 0) + self._rng.randint(1, 4)

        return penalties



    def stats(self) -> dict:
        hospitals = self.hospital_zones()
        total_patients = sum(z.patients_waiting + z.patients_saved + z.patients_lost
                             for z in hospitals)
        saved = sum(z.patients_saved for z in hospitals)
        lost = sum(z.patients_lost for z in hospitals)
        pct = (saved / total_patients * 100) if total_patients > 0 else 0.0
        return {
            "total_patients": total_patients,
            "patients_saved": saved,
            "patients_lost": lost,
            "lives_saved_pct": round(pct, 2),
        }



class BloodBankEnvironment:
    """OpenEnv-compatible async RL environment for blood bank logistics."""

    def __init__(self, scenario_name: str = "city_shortage", rng_seed: int = 42):
        self._scenario_name = scenario_name
        self._rng_seed = rng_seed
        self._grid: Optional[BloodGrid] = None
        self._inventory: Dict[str, int] = {}
        self._capacity: int = 100
        self._step_number: int = 0
        self._max_steps: int = 70
        self._done: bool = False
        self._mission_success: bool = False
        self._episode_start_step: int = 0
        self._total_reward: float = 0.0


    async def reset(self) -> BloodObservation:
        sc = get_scenario(self._scenario_name)
        self._grid = BloodGrid(sc, self._rng_seed)
        self._inventory = dict(sc["initial_inventory"])
        self._capacity = sc["capacity"]
        self._max_steps = sc["max_steps"]
        self._step_number = 0
        self._done = False
        self._mission_success = False
        self._total_reward = 0.0
        return self._obs("Environment reset. Ready to deliver blood.", 0.0, False, False)



    async def step(self, action: DeliveryAction) -> Tuple[BloodObservation, float, bool, dict]:
        if self._done:
            obs = self._obs("Episode already complete.", 0.0, True, self._mission_success)
            return obs, 0.0, True, {"message": "done"}

        self._step_number += 1
        reward = 0.0
        msg = ""

        grid = self._grid

        # ---- Execute action ----
        if action.action_type == ActionType.move:
            if action.direction is None:
                msg = "Move action requires a direction."
            else:
                ok, msg = grid.move_agent(action.direction)
                reward = -0.05 if ok else -0.2  # small step cost

        elif action.action_type == ActionType.deliver:
            if not action.target_zone_id or not action.blood_type or not action.quantity:
                msg = "Deliver action requires target_zone_id, blood_type, and quantity."
                reward = -0.1
            else:
                r, msg, actual = grid.deliver(
                    action.target_zone_id,
                    action.blood_type,
                    action.quantity,
                    self._inventory,
                )
                reward = r
                if actual > 0:
                    self._inventory[action.blood_type] = (
                        self._inventory.get(action.blood_type, 0) - actual
                    )

        elif action.action_type == ActionType.collect:
            if not action.target_zone_id or not action.blood_type or not action.quantity:
                msg = "Collect action requires target_zone_id, blood_type, and quantity."
                reward = -0.1
            else:
                cap_remaining = self._capacity - sum(self._inventory.values())
                qty = min(action.quantity, cap_remaining)
                if qty <= 0:
                    msg = "Inventory full, cannot collect."
                    reward = -0.1
                else:
                    ok, msg, actual = grid.collect(
                        action.target_zone_id,
                        action.blood_type,
                        qty,
                    )
                    if ok and actual > 0:
                        self._inventory[action.blood_type] = (
                            self._inventory.get(action.blood_type, 0) + actual
                        )
                        reward = 0.1  # small positive for restocking
                    else:
                        reward = -0.05

        elif action.action_type == ActionType.wait:
            msg = "Agent waited."
            reward = -0.1  # discourage excessive waiting

        # ---- Advance time ----
        penalties = grid.advance_time()
        penalty_total = sum(penalties.values())
        reward += penalty_total

        self._total_reward += reward

        # ---- Check done ----
        st = grid.stats()
        lives_pct = st["lives_saved_pct"]
        success = lives_pct >= 85.0
        timeout = self._step_number >= self._max_steps

        if success or timeout:
            self._done = True
            self._mission_success = success
            # Terminal reward
            coverage = min(1.0, lives_pct / 100.0)
            speed = max(0.0, 1.0 - self._step_number / self._max_steps)
            terminal = coverage * 6.0 + speed * 2.0 + (10.0 if success else 0.0)
            reward += terminal
            self._total_reward += terminal
            if success:
                msg += f" MISSION COMPLETE! Lives saved: {lives_pct:.1f}%"
            else:
                msg += f" Time limit reached. Lives saved: {lives_pct:.1f}%"

        obs = self._obs(msg, reward, self._done, self._mission_success)
        return obs, reward, self._done, {
            "step": self._step_number,
            "total_reward": self._total_reward,
            "penalties": penalties,
        }

 

    @property
    def state(self) -> dict:
        if self._grid is None:
            return {"status": "not_initialized"}
        st = self._grid.stats()
        ax, ay = self._grid.agent_pos()
        return {
            "scenario": self._scenario_name,
            "step": self._step_number,
            "max_steps": self._max_steps,
            "agent_x": ax,
            "agent_y": ay,
            "inventory": dict(self._inventory),
            "capacity_remaining": self._capacity - sum(self._inventory.values()),
            **st,
            "done": self._done,
            "mission_success": self._mission_success,
        }


    def _obs(self, msg: str, reward: float, done: bool, success: bool) -> BloodObservation:
        grid = self._grid
        ax, ay = grid.agent_pos()
        agent_zone_id = grid.agent_zone_id()
        inv_total = sum(self._inventory.values())
        st = grid.stats()

        agent = AgentStatus(
            current_zone_id=agent_zone_id,
            x=ax,
            y=ay,
            inventory=dict(self._inventory),
            capacity_remaining=self._capacity - inv_total,
            total_units=inv_total,
        )

        blood_bank_ids = [z.zone_id for z in grid.blood_source_zones()
                         if z.zone_type == ZoneType.blood_bank]
        donor_ids = [z.zone_id for z in grid.blood_source_zones()
                     if z.zone_type == ZoneType.donor_center]

        return BloodObservation(
            scenario_name=self._scenario_name,
            step_number=self._step_number,
            max_steps=self._max_steps,
            zones=grid.all_zones(),
            critical_hospitals=grid.critical_hospital_ids(),
            blood_banks=blood_bank_ids,
            donor_centers=donor_ids,
            agent=agent,
            total_patients=st["total_patients"],
            patients_saved=st["patients_saved"],
            patients_lost=st["patients_lost"],
            lives_saved_pct=st["lives_saved_pct"],
            last_action_result=msg,
            last_reward=round(reward, 4),
            is_complete=done,
            mission_success=success,
        )
