#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentState:
    raw_patient: dict
    patient_state: Optional[dict] = None
    first_prediction: Optional[dict] = None
    final_prediction: Optional[dict] = None
    estimated_features: List[str] = field(default_factory=list)
    route: List[str] = field(default_factory=list)
