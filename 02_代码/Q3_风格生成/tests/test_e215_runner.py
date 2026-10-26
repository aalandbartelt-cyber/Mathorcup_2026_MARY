from pathlib import Path
import sys
import unittest

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_e215_strict_eb_migration import validate_config


class E215RunnerTests(unittest.TestCase):
    def test_fixed_geometry_protocol_cannot_change_after_registration(self):
        config = yaml.safe_load((SRC.parent / "configs/E215_strict_eb_migration.yaml").read_text())
        validate_config(config)
        for key, value in [("source_style", "Tripel"), ("mean_kappa", 20.0),
                           ("covariance_kappa", 20.0), ("scenario", "C2"),
                           ("path_lambdas", [0.5, 1.0]), ("q1_retrained", True),
                           ("gpu_used", True), ("sensory_proxy_used_in_geometry_stage", True)]:
            with self.subTest(key=key):
                bad = dict(config, **{key: value})
                with self.assertRaises(ValueError):
                    validate_config(bad)


if __name__ == "__main__":
    unittest.main()
