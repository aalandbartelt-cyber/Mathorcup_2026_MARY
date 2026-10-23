from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

try:
    chemical_evidence = importlib.import_module("chemical_evidence")
except ModuleNotFoundError:
    chemical_evidence = None


class ChemicalEvidenceTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(chemical_evidence, "chemical_evidence module is missing")
        return chemical_evidence

    def test_named_target_direct_identity_is_highest_level(self):
        module = self.require_module()
        evidence = module.semantic_support(
            "X4vg", "X4VG", "4-vinyl guaiacol", "Yeast"
        )
        self.assertEqual(evidence.level, 4)
        self.assertEqual(evidence.label, "direct_identity")

    def test_esters_and_named_descriptors_are_distinguished(self):
        module = self.require_module()
        nail_polish = module.semantic_support(
            "A_esters_ethac", "ethyl_acetate", "Ethyl acetate", "Yeast"
        )
        broad_ester = module.semantic_support(
            "A_esters_all", "ethyl_octanoate", "Ethyl octanoate", "Yeast"
        )
        self.assertEqual(nail_polish.level, 3)
        self.assertEqual(broad_ester.level, 2)

    def test_source_consistency_does_not_turn_yeast_ester_into_hop_compound(self):
        module = self.require_module()
        evidence = module.semantic_support(
            "A_hops_noble", "ethyl_octanoate", "Ethyl octanoate", "Yeast"
        )
        self.assertEqual(evidence.level, 0)

    def test_descriptor_rules_are_specific(self):
        module = self.require_module()
        orange = module.semantic_support("orange", "limonene", "Limonene", "Hops")
        clove = module.semantic_support("clove", "limonene", "Limonene", "Hops")
        self.assertEqual(orange.level, 3)
        self.assertEqual(clove.level, 0)

    def test_methanethiol_is_not_misclassified_as_burnt_malt(self):
        module = self.require_module()
        evidence = module.semantic_support("F_malt_burn", "MeSH", "Methanethiol", "Yeast")
        self.assertEqual(evidence.level, 0)


if __name__ == "__main__":
    unittest.main()
