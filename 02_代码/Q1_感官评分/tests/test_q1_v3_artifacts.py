from __future__ import annotations

import hashlib
import re
import unittest
from colorsys import rgb_to_hsv
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = ROOT / "04_图表" / "Q1_感官评分"
if not (FIGURE_ROOT / "final_v3_draft").is_dir():
    # Historical figures are retained outside the active publication tree.
    FIGURE_ROOT = (ROOT / "_archive/论文交付历史/20261025_最终模型更新前"
                   / "04_图表/Q1_感官评分")
V3_DIR = FIGURE_ROOT / "final_v3_draft"
DATA_DIR = ROOT / "03_结果" / "Q1_感官评分" / "figure_data"
PLOT_SCRIPT = ROOT / "02_代码" / "Q1_感官评分" / "src" / "plot_q1_drafts.R"

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
    def test_v3_model_palette_uses_five_distinct_non_gray_colors(self) -> None:
        source = PLOT_SCRIPT.read_text(encoding="utf-8")
        palette_block = re.search(
            r"model_colors\s*<-\s*c\((.*?)\n\)", source, flags=re.DOTALL
        )
        self.assertIsNotNone(palette_block)
        colors = re.findall(r"#[0-9A-Fa-f]{6}", palette_block.group(1))
        self.assertEqual(len(colors), 5)
        self.assertEqual(len(set(colors)), 5)
        for color in colors:
            red, green, blue = (
                int(color[index:index + 2], 16) / 255
                for index in (1, 3, 5)
            )
            saturation = rgb_to_hsv(red, green, blue)[1]
            self.assertGreater(saturation, 0.20, f"near-gray model color: {color}")

    def test_v3_figure2_is_three_panel_r2_spearman_atlas(self) -> None:
        source = PLOT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("target_panel_plot <- function", source)
        self.assertIn('metric = "Spearman"', source)
        self.assertIn("plot_layout(widths = c(1, 1, 1)", source)
        self.assertNotIn("（15 项）", source)
        self.assertNotIn("（20 项）", source)

    def test_v3_figure4_global_mae_label_is_horizontal(self) -> None:
        source = PLOT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('label = sprintf("总体 MAE=%.3f"', source)
        self.assertNotIn("angle = 90", source)

    def test_v3_figure1_uses_plain_numeric_extratrees_labels(self) -> None:
        source = PLOT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('subtitle = "（a）解释度"', source)
        self.assertIn('subtitle = "（b）绝对误差"', source)
        self.assertIn('label = sprintf("%.3f", r2)', source)
        self.assertIn('label = sprintf("%.3f", mae)', source)
        self.assertNotIn("解释度：ExtraTrees", source)
        self.assertNotIn("绝对误差：ExtraTrees", source)

    def test_v3_figure4_mapping_uses_bottom_legend_not_direct_labels(self) -> None:
        source = PLOT_SCRIPT.read_text(encoding="utf-8")
        figure4b = source.split("p4b <-", maxsplit=1)[1].split(
            "p4 <-", maxsplit=1
        )[0]
        self.assertNotIn("geom_text(", figure4b)
        self.assertNotIn("mapping_labels", figure4b)
        self.assertIn("guide_legend(nrow = 2", figure4b)
        self.assertIn('legend.position = "bottom"', figure4b)

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
        figure2 = manifest.loc[manifest["figure_no"].astype(str) == "2"].iloc[0]
        self.assertIn("Spearman", figure2["caption"])
        self.assertNotIn("目标分布", figure2["caption"])

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
