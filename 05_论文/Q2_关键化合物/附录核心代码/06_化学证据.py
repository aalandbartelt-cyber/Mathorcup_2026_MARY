"""Conservative, auditable semantic evidence rules for Q2 candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticEvidence:
    level: int
    label: str
    reason: str
    literature_code: str


DIRECT = {
    "X4vg": {"X4VG"},
    "diacetyl": {"X2.3.butanedione"},
    "dms": {"DMS"},
    "t2n": {"X.E..2.nonenal"},
    "alcohol": {"ethanol..v.v."},
    "body": {"ethanol..v.v."},
    "co2": {"CO2.PSI."},
    "lactic": {"lactic_acid.mg.L."},
    "acetic": {"acetic_acid.g.L."},
}

ESTER_TOKENS = (
    "acetate",
    "formate",
    "butyrate",
    "propionate",
    "hexanoate",
    "octanoate",
    "decanoate",
    "succinate",
    "tiglate",
    "isovalerate",
    "pentanoate",
)

TERPENE_CITRUS = {
    "limonene",
    "linalool",
    "geraniol",
    "citronellol",
    "X1.alpha.terpineol",
    "ethyl..E..linalool",
    "methyl.geranate",
}
WOODY = (
    "caryophyll",
    "humulene",
    "cedr",
    "eudesmol",
    "cubenol",
    "calacorene",
)
MALT_BREAD = ("furfural", "acetylfuran", "furanmethanol")
MALT_CARAMEL = ("furfural", "isomaltol", "maltol", "furaneol")
MALT_BURNT = ("pyrazine", "furfural", "guaiacol")


def _evidence(level: int, reason: str, literature: str) -> SemanticEvidence:
    labels = {
        4: "direct_identity",
        3: "specific_descriptor",
        2: "source_or_class_consistent",
        1: "broad_relation",
        0: "no_specific_support",
    }
    return SemanticEvidence(level, labels[level], reason, literature)


def semantic_support(
    target_code: str,
    feature_code: str,
    chemical_name: str,
    origin_paper: str,
) -> SemanticEvidence:
    """Return a conservative semantic match independent of model scores."""
    target = str(target_code)
    feature = str(feature_code)
    name = str(chemical_name).lower()
    origin = str(origin_paper)
    if feature in DIRECT.get(target, set()):
        return _evidence(4, "感官维度与测量化合物为同名直接锚点", "C1;C4")

    is_ester = any(token in feature.lower() or token in name for token in ESTER_TOKENS)
    if target.endswith("esters_ethac") and feature == "ethyl_acetate":
        return _evidence(3, "乙酸乙酯具有果香及高浓度溶剂/指甲油样气味", "C2;C3")
    if target.endswith("esters_isoaa") and feature == "isopentyl_acetate":
        return _evidence(3, "乙酸异戊酯是典型香蕉样酯香锚点", "C2;C3")
    if "esters" in target and is_ester:
        return _evidence(2, "候选属于啤酒中常见的挥发性酯类", "C2;C3;C6")

    if target.startswith(("A_hops_", "F_hops_")):
        if target.endswith("citrus") and feature in TERPENE_CITRUS:
            return _evidence(3, "单萜类与柑橘/花香型酒花香气相符", "C2")
        if target.endswith("woody") and any(token in feature for token in WOODY):
            return _evidence(3, "倍半萜/含氧萜与木质酒花香气相符", "C2")
        if origin == "Hops":
            return _evidence(2, "附件 8 将该化合物来源标为酒花", "C1;C2")
        return _evidence(0, "未找到面向该酒花描述词的特异化学支持", "C1")

    if target.startswith(("A_malt_", "F_malt_")):
        token_sets = []
        if target.endswith("bread"):
            token_sets = MALT_BREAD
        elif target.endswith("cara"):
            token_sets = MALT_CARAMEL
        elif target.endswith("burn"):
            token_sets = MALT_BURNT
        if token_sets and any(token.lower() in feature.lower() for token in token_sets):
            return _evidence(3, "呋喃/吡嗪等麦芽热反应产物与该描述词相符", "C3;C4")
        if origin == "Malt":
            return _evidence(2, "附件 8 将该化合物来源标为麦芽", "C1;C3")
        return _evidence(0, "未找到面向该麦芽描述词的特异化学支持", "C1")

    if target == "orange" and feature in TERPENE_CITRUS:
        return _evidence(3, "柠檬烯/芳樟醇类单萜与橙柑香相符", "C2")
    if target == "coriander" and feature in TERPENE_CITRUS:
        return _evidence(3, "芳樟醇与相关单萜是芫荽籽香气的重要组成", "C2")
    if target == "clove" and any(
        token in feature.lower() for token in ("x4vg", "eugenol", "isoeugenol", "guaiacol")
    ):
        return _evidence(3, "挥发性酚/丁香酚具有丁香与辛香描述", "C4;C5")
    if target == "acidity" and feature in {"lactic_acid.mg.L.", "acetic_acid.g.L."}:
        return _evidence(3, "乳酸或乙酸可直接贡献酸味与酸度", "C3")
    if target == "body" and feature == "glycerol":
        return _evidence(3, "甘油与口感丰满度具有直接的理化联系", "C3")
    if target == "sweetness" and feature == "glycerol":
        return _evidence(3, "甘油兼具甜感与口感贡献", "C3")
    if target == "stale_hops" and feature == "X.E..2.nonenal":
        return _evidence(4, "trans-2-nonenal 是纸板样啤酒陈化锚点", "C4")
    if target == "barnyard" and any(
        token in feature.lower() for token in ("ethyl.phenol", "ethylguaiacol", "x4eg")
    ):
        return _evidence(3, "Brett 相关挥发性酚具有马厩/皮革描述", "C5")
    if target in {"aftertaste", "overall"}:
        return _evidence(1, "综合感官终点允许多类香味化合物形成间接关系", "C1")
    return _evidence(0, "附件与核验文献未给出该候选—描述词的特异联系", "C1")
