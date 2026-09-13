#!/usr/bin/env python3
from pathlib import Path

import yaml


DEFAULT_PATH = Path("/public_bme/home/jiawei2022/tongren_bme_transition/code/src/config/clinical_reference.yaml")


class ClinicalReferenceMemory:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.data.get(key, default)
