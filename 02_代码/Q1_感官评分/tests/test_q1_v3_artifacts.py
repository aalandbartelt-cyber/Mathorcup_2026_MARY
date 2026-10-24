from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = ROOT / "04_图表" / "Q1_感官评分"
V3_DIR = FIGURE_ROOT / "final_v3_draft"
DATA_DIR = ROOT / "03_结果" / "Q1_感官评分" / "figure_data"

EXPECTED_IMAGES = {
    "图1_模型对比.png": (2460, 1305),
    "图2_目标分布与逐目标R2.png": (3360, 1890),
    "图3_OOF观测预测.png": (2640, 1260),
    "图4_风格误差与量表映射.png": (3150, 1770),
    "附图S1_残差诊断.png": (2640, 1260),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksum_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checksum, filename = line.split(maxsplit=1)
        entries[filename.strip()] = checksum
    return entries


class Q1V3ArtifactTests(unittest.TestCase):
    def test_v3_directory_contains_expected_images(self) -> None:
        self.assertTrue(V3_DIR.is_dir(), f"missing V3 directory: {V3_DIR}")
        actual = {path.name for path in V3_DIR.glob("*.png")}
        self.assertEqual(actual, set(EXPECTED_IMAGES))

    def test_v3_dimensions_and_resolution(self) -> None:
        for filename, expected_size in EXPECTED_IMAGES.items():
            with self.subTest(filename=filename):
                with Image.open(V3_DIR / filename) as image:
                    self.assertEqual(image.size, expected_size)
                    dpi = image.info.get("dpi")
                    self.assertIsNotNone(dpi)
                    self.assertAlmostEqual(dpi[0], 300.0, delta=1.0)
                    self.assertAlmostEqual(dpi[1], 300.0, delta=1.0)

    def test_v3_figure_list_has_four_main_and_one_supplementary(self) -> None:
        manifest = pd.read_csv(V3_DIR / "Q1_图清单.csv")
        self.assertEqual(len(manifest), 5)
        self.assertEqual((manifest["role"] == "main").sum(), 4)
        self.assertEqual((manifest["role"] == "supplementary").sum(), 1)
        self.assertTrue(manifest["path"].str.contains("final_v3_draft", regex=False).all())

    def test_v3_checksum_manifest_matches_artifacts(self) -> None:
        manifest_path = V3_DIR / "SHA256SUMS.txt"
        self.assertTrue(manifest_path.is_file())
        entries = read_checksum_manifest(manifest_path)
        expected_names = set(EXPECTED_IMAGES) | {"Q1_图清单.csv"}
        self.assertEqual(set(entries), expected_names)
        for filename, expected_hash in entries.items():
            self.assertEqual(sha256(V3_DIR / filename), expected_hash)

    def test_frozen_v1_and_v2_checksum_manifests_remain_valid(self) -> None:
        for directory_name in ("final_v1", "final_v2_draft"):
            directory = FIGURE_ROOT / directory_name
            entries = read_checksum_manifest(directory / "SHA256SUMS.txt")
            self.assertGreaterEqual(len(entries), 6)
            for filename, expected_hash in entries.items():
                with self.subTest(version=directory_name, filename=filename):
                    self.assertEqual(sha256(directory / filename), expected_hash)

    def test_frozen_source_contract_counts(self) -> None:
        comparison = pd.read_csv(DATA_DIR / "model_comparison.csv")
        targets = pd.read_csv(DATA_DIR / "target_metrics_extra_trees.csv")
        styles = pd.read_csv(DATA_DIR / "style_metrics_extra_trees.csv")
        representatives = pd.read_csv(DATA_DIR / "representative_oof.csv.gz")

        self.assertEqual(comparison.groupby("display_name").size().to_dict(), {
            "ExtraTrees": 10,
            "PLS2（秩高斯）": 10,
            "Ridge（秩高斯）": 10,
            "全局均值": 10,
            "风格收缩均值": 10,
        })
        self.assertEqual(targets["target_code"].nunique(), 50)
        self.assertEqual(styles["tasting_category_fine"].nunique(), 22)
        self.assertTrue(
            {"acidity", "X4vg", "stale_hops"}.issubset(
                set(representatives["target_code"])
            )
        )
        self.assertAlmostEqual(float(targets["mae"].mean()), 0.338879, places=6)


if __name__ == "__main__":
    unittest.main()
