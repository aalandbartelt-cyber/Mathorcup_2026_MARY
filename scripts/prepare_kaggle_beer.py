#!/usr/bin/env python3
"""Prepare the Kaggle Top Beer Information data for later external style validation.

This script does NOT train or modify any Q1/Q2/Q3 model. It produces:
  processed/kaggle_beer_clean.csv
  processed/kaggle_style_profiles.csv
  processed/kaggle_official_style_mapping_draft.csv
  processed/kaggle_official_style_profiles_draft.csv
  processed/kaggle_quality_report.md

The official-style mapping is a provisional draft. Freeze it before comparing it
with any model output, and never use these rows as chemical training samples.
"""
from __future__ import annotations

from pathlib import Path

import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "2026初赛" / "公开数据" / "Kaggle_top_beer_information"
OUT_DIR = RAW_DIR / "processed"

FLAVORS = [
    "Astringency", "Body", "Alcohol", "Bitter", "Sweet",
    "Sour", "Salty", "Fruits", "Hoppy", "Spices", "Malty",
]

# (official style, kaggle style, subset, confidence, primary_pool, rationale)
# primary_pool=1 rows form the draft official-style profile; 0 rows are
# sensitivity candidates and must not be silently pooled into the primary result.
MAPPING: list[tuple[str, str, str, str, int, str]] = [
    ("Blond", "Blonde Ale - Belgian", "", "high", 1, "Belgian blonde is the closest Kaggle substyle"),
    ("Blond", "Blonde Ale - American", "", "medium", 0, "American blonde is a broader pale-ale fallback"),
    ("Tripel", "Tripel", "", "high", 1, "direct style match"),
    ("Strong ale", "Strong Ale - Belgian Dark", "", "high", 1, "Belgian strong dark"),
    ("Strong ale", "Strong Ale - Belgian Pale", "", "medium", 1, "Belgian strong pale"),
    ("Strong ale", "Strong Ale - American", "", "medium", 1, "broad strong-ale family"),
    ("Strong ale", "Strong Ale - English", "", "medium", 1, "broad strong-ale family"),
    ("Hoppy", "IPA - American", "", "high", 1, "canonical hoppy style"),
    ("Hoppy", "IPA - English", "", "medium", 1, "hoppy ale family"),
    ("Hoppy", "IPA - Imperial", "", "medium", 1, "hoppy ale family"),
    ("Hoppy", "IPA - New England", "", "medium", 1, "hoppy ale family"),
    ("Hoppy", "Pale Ale - American", "", "medium", 1, "hoppy pale ale"),
    ("Hoppy", "Pale Ale - English", "", "medium", 1, "hoppy pale ale"),
    ("Hoppy", "Pale Ale - Belgian", "", "low", 0, "Belgian pale ale is less hop-led"),
    ("Stout/Porter", "Stout - American Imperial", "", "medium", 1, "stout family"),
    ("Stout/Porter", "Stout - American", "", "medium", 1, "stout family"),
    ("Stout/Porter", "Stout - Irish Dry", "", "high", 1, "canonical stout"),
    ("Stout/Porter", "Stout - Oatmeal", "", "medium", 1, "stout family"),
    ("Stout/Porter", "Stout - Foreign / Export", "", "medium", 1, "stout family"),
    ("Stout/Porter", "Stout - English", "", "high", 1, "canonical stout"),
    ("Stout/Porter", "Stout - Sweet / Milk", "", "medium", 1, "stout family"),
    ("Stout/Porter", "Stout - Russian Imperial", "", "medium", 1, "stout family"),
    ("Stout/Porter", "Porter - English", "", "high", 1, "canonical porter"),
    ("Stout/Porter", "Porter - Imperial", "", "medium", 1, "porter family"),
    ("Stout/Porter", "Porter - Robust", "", "medium", 1, "porter family"),
    ("Stout/Porter", "Porter - Smoked", "", "medium", 1, "porter family"),
    ("Stout/Porter", "Porter - American", "", "medium", 1, "porter family"),
    ("Stout/Porter", "Porter - Baltic", "", "medium", 1, "porter family"),
    ("Brown", "Brown Ale - American", "", "high", 1, "brown ale family"),
    ("Brown", "Brown Ale - Belgian Dark", "", "medium", 1, "brown ale family"),
    ("Brown", "Brown Ale - English", "", "high", 1, "canonical brown ale"),
    ("Amber", "Red Ale - American Amber / Red", "", "high", 1, "amber/red ale"),
    ("Amber", "Lager - American Amber / Red", "", "medium", 1, "amber lager"),
    ("Amber", "Altbier", "", "low", 0, "amber-coloured but top-fermented altbier"),
    ("Wheat", "Wheat Beer - American Pale", "", "medium", 1, "wheat beer family"),
    ("Wheat", "Wheat Beer - Dunkelweizen", "", "high", 1, "wheat beer family"),
    ("Wheat", "Wheat Beer - Hefeweizen", "", "high", 1, "canonical wheat beer"),
    ("Wheat", "Wheat Beer - Kristallweizen", "", "medium", 1, "wheat beer family"),
    ("Wheat", "Wheat Beer - Witbier", "", "high", 1, "witbier is a wheat style"),
    ("Wheat", "Wheat Beer - American Dark", "", "medium", 1, "wheat beer family"),
    ("Wheat", "Wheat Beer - Wheatwine", "", "low", 0, "strong barleywine-like wheat outlier"),
    ("Lambic", "Lambic - Traditional", "", "high", 1, "traditional lambic"),
    ("Lambic", "Lambic - Gueuze", "", "high", 1, "gueuze is blended lambic"),
    ("Pils/Lager", "Pilsner - Bohemian / Czech", "", "high", 1, "canonical pilsner"),
    ("Pils/Lager", "Pilsner - German", "", "high", 1, "canonical pilsner"),
    ("Pils/Lager", "Lager - American", "", "medium", 1, "lager family"),
    ("Pils/Lager", "Lager - European Pale", "", "medium", 1, "lager family"),
    ("Pils/Lager", "Lager - Helles", "", "medium", 1, "pale lager family"),
    ("Pils/Lager", "Lager - European Dark", "", "medium", 1, "dark lager family"),
    ("Pils/Lager", "Lager - Munich Dunkel", "", "medium", 1, "dark lager family"),
    ("Pils/Lager", "Lager - Märzen / Oktoberfest", "", "medium", 1, "amber lager family"),
    ("Pils/Lager", "Lager - Schwarzbier", "", "medium", 1, "dark lager family"),
    ("Pils/Lager", "Lager - Vienna", "", "medium", 1, "amber lager family"),
    ("Pils/Lager", "Lager - Kellerbier / Zwickelbier", "", "medium", 1, "unfiltered lager family"),
    ("Pils/Lager", "Pilsner - Imperial", "", "low", 0, "strong pilsner outlier"),
    ("Pils/Lager", "Lager - European Strong", "", "low", 0, "strong lager outlier"),
    ("Pils/Lager", "Lager - Adjunct", "", "low", 0, "adjunct lager is a broad fallback"),
    ("Pils/Lager", "Lager - Light", "", "low", 0, "light lager is a low-calorie fallback"),
    ("Pils/Lager", "Lager - Malt Liquor", "", "low", 0, "malt liquor is a separate product"),
    ("Pils/Lager", "Lager - Japanese Rice", "", "low", 0, "rice adjunct lager"),
    ("Pils/Lager", "Lager - India Pale Lager (IPL)", "", "low", 0, "hop-forward hybrid"),
    ("Pils/Lager", "Lager - Rauchbier", "", "low", 0, "smoked lager outlier"),
    ("Pils/Lager", "California Common / Steam Beer", "", "low", 0, "hybrid lager"),
    ("Dubbel", "Dubbel", "", "high", 1, "direct style match"),
    ("Saison", "Farmhouse Ale - Saison", "", "high", 1, "direct style match"),
    ("Saison", "Farmhouse Ale - Bière de Garde", "", "medium", 0, "related farmhouse ale"),
    ("Kriek", "Lambic - Fruit (Kriek/Cherry/Cerise)", "kriek", "high", 1, "name/description-matched Kriek subset"),
    ("Kriek", "Lambic - Fruit", "", "medium", 0, "all fruit lambics; broader sensitivity pool"),
    ("Christmas", "Winter Warmer", "", "high", 1, "seasonal winter warmer"),
    ("Christmas", "Herb and Spice Beer", "", "medium", 0, "spiced holiday-beer fallback"),
    ("Low/No alcohol", "Low Alcohol Beer", "", "high", 1, "direct low-alcohol category"),
    ("Low/No alcohol", "Lager - Light", "", "medium", 0, "light lager fallback"),
    ("Fruitbeer", "Fruit and Field Beer", "", "high", 1, "direct fruit-beer category"),
    ("Fruitbeer", "Lambic - Fruit", "", "medium", 0, "fruit lambic fallback; Kriek kept separate"),
    ("West Flanders ale", "Sour - Flanders Red Ale", "", "high", 1, "Flanders red ale"),
    ("West Flanders ale", "Sour - Flanders Oud Bruin", "", "medium", 0, "Flanders brown fallback"),
    ("Brett/cofermented", "Brett Beer", "", "high", 1, "direct Brett category"),
    ("Brett/cofermented", "Wild Ale", "", "medium", 0, "wild/mixed-fermentation fallback"),
    ("Flanders old brown", "Sour - Flanders Oud Bruin", "", "high", 1, "Oud Bruin is the direct match"),
    ("Scotch", "Scotch Ale / Wee Heavy", "", "high", 1, "direct Scotch ale match"),
    ("Scotch", "Scottish Ale", "", "medium", 0, "Scottish ale fallback"),
    ("Brut", "Bière de Champagne / Bière Brut", "", "high", 1, "bière brut"),
    ("Brut", "IPA - Brut", "", "medium", 0, "brut IPA fallback"),
    ("Faro", "Lambic - Faro", "", "high", 1, "direct Faro match"),
]

# Kaggle has 11 coarse flavour word-frequency dimensions; the official data has
# 50 specific targets. Only these proxies are defensible. Salty and Astringency
# have no clean official counterpart and are deliberately left unused.
# (kaggle_dimension, official_primary, official_secondary, confidence, note)
PROXY: list[tuple[str, str, str, str, str]] = [
    ("Body", "body", "", "high", "direct target"),
    ("Alcohol", "alcohol", "", "high", "direct target"),
    ("Bitter", "bitternes", "", "high", "direct target; official spelling"),
    ("Sweet", "sweetness", "", "high", "direct target"),
    ("Sour", "acidity", "lactic | acetic", "high", "acidity primary; lactic/acetic sensitivity"),
    ("Hoppy", "F_hops_all", "A_hops_all", "high", "flavour overall primary; aroma sensitivity"),
    ("Malty", "F_malt_all", "A_malt_all", "high", "flavour overall primary; aroma sensitivity"),
    ("Fruits", "F_esters_fruity", "A_esters_fruity | orange", "medium", "coarse fruity proxy"),
    ("Spices", "clove | coriander", "", "medium", "no overall spice target; average clove/coriander"),
    ("Astringency", "", "", "none", "no clean official target; do not use"),
    ("Salty", "", "", "none", "no official salty target; do not use"),
]


def load_raw() -> pd.DataFrame:
    path = RAW_DIR / "beer_data_set.csv"
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode {path}")


def build_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Name"] = out["Name"].fillna("")
    out["name_missing"] = out["Name"].eq("")
    out["name_duplicate"] = out["Name"].duplicated(keep=False) & ~out["name_missing"]

    for col in FLAVORS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["flavor_total"] = out[FLAVORS].sum(axis=1)
    out["flavor_usable"] = out["flavor_total"] > 0
    for col in FLAVORS:
        out[f"prop_{col}"] = np.where(
            out["flavor_usable"], out[col] / out["flavor_total"], np.nan
        )

    out["abv_valid"] = out["ABV"].gt(0) & out["ABV"].le(20)
    out["abv_flag"] = np.select(
        [out["ABV"].le(0), out["ABV"].gt(20)], ["nonpositive", "high"], default="ok"
    )
    out["rating_valid"] = out["Ave Rating"].gt(0) & out["Ave Rating"].le(5)

    text = (out["Name"].astype(str) + " " + out["Description"].astype(str)).str.lower()
    out["kriek_subset"] = out["Style"].eq("Lambic - Fruit") & text.str.contains(
        "kriek|cherry|cerise", regex=True, na=False
    )
    return out


def _profile(rows: pd.DataFrame) -> dict:
    usable = rows[rows["flavor_usable"]]
    valid_abv = rows[rows["abv_valid"]]
    valid_rating = rows[rows["rating_valid"]]
    rec: dict = {
        "n_total": int(len(rows)),
        "n_flavor_usable": int(len(usable)),
        "abv_median": float(valid_abv["ABV"].median()) if len(valid_abv) else np.nan,
        "rating_mean": float(valid_rating["Ave Rating"].mean()) if len(valid_rating) else np.nan,
    }
    for col in FLAVORS:
        rec[f"mean_prop_{col}"] = float(usable[f"prop_{col}"].mean()) if len(usable) else np.nan
    top = usable[[f"prop_{c}" for c in FLAVORS]].mean().sort_values(ascending=False)
    rec["top_flavor_1"] = top.index[0].removeprefix("prop_") if len(top) else ""
    rec["top_flavor_2"] = top.index[1].removeprefix("prop_") if len(top) > 1 else ""
    return rec


def build_profiles(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for style, group in clean.groupby("Style", sort=True):
        rows.append({"kaggle_style": style, "subset": "", **profile_row(group)})
    kriek = clean[clean["kriek_subset"]]
    rows.append({
        "kaggle_style": "Lambic - Fruit (Kriek/Cherry/Cerise)",
        "subset": "kriek",
        **profile_row(kriek),
    })
    return pd.DataFrame(rows)


def profile_row(rows: pd.DataFrame) -> dict:
    rec = _profile(rows)
    return rec


def build_mapping(profiles: pd.DataFrame) -> pd.DataFrame:
    available = set(profiles["kaggle_style"])
    missing = sorted({m[1] for m in MAPPING} - available)
    if missing:
        raise ValueError(f"Mapping references missing Kaggle styles: {missing}")
    rows = []
    for official, kaggle, subset, confidence, primary, rationale in MAPPING:
        rows.append({
            "official_style": official,
            "kaggle_style": kaggle,
            "subset": subset,
            "confidence": confidence,
            "primary_pool": int(primary),
            "rationale": rationale,
            "mapping_status": "draft_freeze_before_validation",
        })
    return pd.DataFrame(rows)


def build_proxy() -> pd.DataFrame:
    rows = []
    for kaggle_dim, primary, secondary, confidence, note in PROXY:
        rows.append({
            "kaggle_dimension": kaggle_dim,
            "official_target_primary": primary,
            "official_target_secondary": secondary,
            "confidence": confidence,
            "usage_status": "use" if primary else "do_not_use",
            "note": note,
            "mapping_status": "draft_freeze_before_validation",
        })
    return pd.DataFrame(rows)


def build_official_profiles(clean: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for official, m in mapping[mapping["primary_pool"].eq(1)].groupby("official_style", sort=False):
        pieces = []
        for _, r in m.iterrows():
            if r["subset"] == "kriek":
                piece = clean[clean["kriek_subset"]]
            else:
                piece = clean[clean["Style"].eq(r["kaggle_style"])]
            pieces.append(piece)
        pool = pd.concat(pieces, ignore_index=True) if pieces else clean.iloc[0:0]
        rec = _profile(pool)
        rec.update({
            "official_style": official,
            "mapping_n_kaggle_styles": int(len(m)),
            "mapping_kaggle_styles": " | ".join(m["kaggle_style"].tolist()),
            "mapping_status": "draft_freeze_before_validation",
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def quality_report(df: pd.DataFrame, clean: pd.DataFrame, profiles: pd.DataFrame,
                   mapping: pd.DataFrame, official_profiles: pd.DataFrame,
                   proxy: pd.DataFrame, raw_hash: str) -> str:
    flavor_zero = int((~clean["flavor_usable"]).sum())
    abv_bad = clean["abv_flag"].value_counts().to_dict()
    usable_proxy = proxy[proxy["usage_status"].eq("use")]
    lines = [
        "# Kaggle Top Beer Information 数据质量与预处理报告",
        "",
        "> 本报告只做外部风格校验的数据准备，不进入 Q1/Q2/Q3 任何模型或论文主结果。",
        "",
        "## 1. 原始数据",
        f"- 文件：`{RAW_DIR / 'beer_data_set.csv'}`",
        f"- SHA-256：`{raw_hash}`",
        f"- 规模：{len(df)} 行 × {df.shape[1]} 列；Kaggle 风格数：{clean['Style'].nunique()}",
        f"- 字段：{', '.join(df.columns)}",
        f"- 缺失：Name {int(df['Name'].isna().sum())}；其余字段 0 缺失",
        f"- 重复 key：{int(df['key'].duplicated().sum())}；重复 Name：{int(df['Name'].duplicated().sum())}（key 唯一，保留行并标记）",
        "",
        "## 2. 清洗与标记",
        f"- `flavor_total` = 11 个风味词频之和；`flavor_usable=False` 的行：{flavor_zero}",
        "- 风味比例为行内 11 列归一化，仅在 `flavor_usable=True` 时计算；全零行保留但不进入风格画像。",
        f"- ABV 标记：{abv_bad}；`abv_valid` 定义为 0 < ABV ≤ 20，仅用于 ABV 汇总。",
        f"- Kriek/Cherry/Cerise 子集：{int(clean['kriek_subset'].sum())} 行（仅从 `Lambic - Fruit` 的名称/描述匹配）。",
        "- 官方附件没有啤酒名称/酒厂，Kaggle `key` 与官方 `beer_id` 不可连接；不合并训练样本。",
        "",
        "## 3. 风格画像",
        f"- `kaggle_style_profiles.csv`：{len(profiles)} 行（112 个 Kaggle 风格 + Kriek 子集）。",
        "- 每行包含 n_total、n_flavor_usable、11 个 `mean_prop_*`、ABV 中位数、评分均值、Top2 风味。",
        "",
        "## 4. 官方 22 风格映射草案",
        f"- `kaggle_official_style_mapping_draft.csv`：{len(mapping)} 条候选；primary_pool=1 共 {int(mapping['primary_pool'].sum())} 条。",
        "- 置信度 high/medium/low 只是映射先验；使用前必须冻结，冻结后不得根据模型结果改映射。",
        f"- `kaggle_official_style_profiles_draft.csv`：{len(official_profiles)} 个官方风格，按 primary_pool 汇总。",
        "",
        "## 5. 感官代理草案（11 → 可用 9 维）",
        f"- `kaggle_sensory_proxy_draft.csv`：11 个 Kaggle 维度，其中 {len(usable_proxy)} 个可进入外部校验。",
        "- Salty 与 Astringency 没有干净的官方目标，标记 do_not_use，不得强行映射。",
        "- Spices 用 clove/coriander 平均作为代理；Fruits 用 F_esters_fruity 为主、A_esters_fruity/orange 作敏感性。",
        "",
        "## 6. 使用边界",
        "- 只能做风格级外部合理性/外部效度校验，不能当化学浓度、专家 0–6 评分或因果证据。",
        "- 不能把 5,558 行拼进 250 个官方样本；不能做啤酒实体级连接。",
        "- 论文可引用 Kaggle 来源；支撑材料不要重新分发原始数据。",
        "- 正式校验前先冻结映射、代理和风味列定义；报告样本量、缺失和映射敏感性。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    clean = build_clean(raw)
    profiles = build_profiles(clean)
    mapping = build_mapping(profiles)
    proxy = build_proxy()
    official = build_official_profiles(clean, mapping)
    mapping["subset"] = mapping["subset"].fillna("")
    profiles["subset"] = profiles["subset"].fillna("")
    raw_hash = hashlib.sha256((RAW_DIR / "beer_data_set.csv").read_bytes()).hexdigest()

    clean.to_csv(OUT_DIR / "kaggle_beer_clean.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(OUT_DIR / "kaggle_style_profiles.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(OUT_DIR / "kaggle_official_style_mapping_draft.csv", index=False, encoding="utf-8-sig")
    proxy.to_csv(OUT_DIR / "kaggle_sensory_proxy_draft.csv", index=False, encoding="utf-8-sig")
    official.to_csv(OUT_DIR / "kaggle_official_style_profiles_draft.csv", index=False, encoding="utf-8-sig")
    report = quality_report(raw, clean, profiles, mapping, official, proxy, raw_hash)
    (OUT_DIR / "kaggle_quality_report.md").write_text(report, encoding="utf-8")

    print(f"raw={raw.shape} clean={clean.shape} profiles={profiles.shape} "
          f"mapping={mapping.shape} proxy={proxy.shape} official={official.shape}")
    print(f"flavor_usable={int(clean['flavor_usable'].sum())} "
          f"all_zero={int((~clean['flavor_usable']).sum())} "
          f"kriek_subset={int(clean['kriek_subset'].sum())}")
    print(f"output={OUT_DIR}")


if __name__ == "__main__":
    main()
