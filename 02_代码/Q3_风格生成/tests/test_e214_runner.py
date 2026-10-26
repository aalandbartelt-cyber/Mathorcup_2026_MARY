from pathlib import Path
import sys
import unittest

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class E214RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(__import__("importlib.util").util.find_spec("run_e214_floor_aware_copula"))
        from run_e214_floor_aware_copula import validate_formal_config
        self.validate = validate_formal_config
        self.config = yaml.safe_load((SRC.parent / "configs/E214_floor_aware_copula.yaml").read_text())

    def test_protocol_requires_registered_floor_and_frozen_cv(self):
        self.validate(self.config)
        for key, value in [("n_repeats", 3), ("n_folds", 2), ("gpu_used", True), ("q1_retrained", True), ("nnls_refit", True)]:
            with self.subTest(key=key):
                bad = dict(self.config)
                bad[key] = value
                with self.assertRaises(ValueError):
                    self.validate(bad)
        bad = dict(self.config)
        bad["floor_fields"] = {"some_other_field": self.config["floor_fields"]["X4VG"]}
        with self.assertRaises(ValueError):
            self.validate(bad)

    def test_only_registered_candidate_and_five_streams_are_allowed(self):
        bad = dict(self.config, methods=["style_copula_shrunk"])
        with self.assertRaises(ValueError):
            self.validate(bad)
        bad = dict(self.config, seeds=[1, 2, 3, 4])
        with self.assertRaises(ValueError):
            self.validate(bad)


if __name__ == "__main__":
    unittest.main()
