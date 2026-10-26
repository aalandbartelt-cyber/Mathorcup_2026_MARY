from pathlib import Path
import sys
import unittest

import pandas as pd
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_e215_strict_eb_migration import validate_config, validate_geometry_outputs


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

    def test_geometry_checkpoint_requires_exact_pairs_and_source_identity(self):
        config = yaml.safe_load((SRC.parent / "configs/E215_strict_eb_migration.yaml").read_text())
        beers = list(range(11))
        endpoints = pd.DataFrame([
            {"beer": beer, "target_style": style, "n_direct_failed": 0, "n_direct_unknown": 1}
            for beer in beers for style in config["target_styles"]
        ])
        paths = pd.DataFrame([
            {"beer": beer, "target_style": style, "lambda": value, "source_identity": value == 0}
            for beer in beers for style in config["target_styles"]
            for value in [0.0] + config["path_lambdas"]
        ])
        validate_geometry_outputs(endpoints, paths, config)
        with self.assertRaises(ValueError):
            validate_geometry_outputs(endpoints.iloc[:-1], paths, config)
        bad = paths.copy()
        bad.loc[bad["lambda"] == 0.0, "source_identity"] = False
        with self.assertRaises(ValueError):
            validate_geometry_outputs(endpoints, bad, config)


if __name__ == "__main__":
    unittest.main()
