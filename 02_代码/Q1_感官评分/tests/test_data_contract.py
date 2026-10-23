from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_contract import (  # noqa: E402
    compound_name_map,
    load_q1,
    validate_compound_name_map,
)


class DataContractTests(unittest.TestCase):
    def test_official_compound_name_map_is_complete_and_one_to_one(self) -> None:
        x, _, _ = load_q1()
        mapping = compound_name_map(x)
        audit = validate_compound_name_map(mapping, x)
        self.assertEqual(audit["rows"], 231)
        self.assertEqual(audit["unique_compound_names"], 231)
        self.assertEqual(audit["unique_feature_codes"], 231)
        self.assertEqual(audit["attachment8_field_codes_matched"], 231)
        self.assertEqual(audit["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
