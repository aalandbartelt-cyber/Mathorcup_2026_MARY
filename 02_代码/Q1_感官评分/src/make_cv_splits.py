"""Create the frozen repeated style-stratified outer folds for Q1."""

from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from data_contract import ATTACHMENTS, RESULT_DIR, load_q1


N_SPLITS = 3
N_REPEATS = 10
RANDOM_STATE = 20261023


def main() -> None:
    x, _, style = load_q1()
    chemistry = pd.read_csv(ATTACHMENTS / "1.csv")
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    rows = []
    for split_position, (_, validation_index) in enumerate(splitter.split(x, style)):
        repeat = split_position // N_SPLITS
        fold = split_position % N_SPLITS
        for index in validation_index:
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "split_id": f"R{repeat:02d}F{fold}",
                    "row_position": int(index),
                    "beer": int(chemistry.iloc[index]["beer"]),
                    "tasting_category_fine": style.iloc[index],
                }
            )

    splits = pd.DataFrame(rows).sort_values(["repeat", "fold", "row_position"])
    counts = splits.groupby(["repeat", "beer"]).size()
    if len(splits) != len(x) * N_REPEATS or not counts.eq(1).all():
        raise RuntimeError("each beer must occur in one validation fold per repeat")

    rare_style = style.value_counts().idxmin()
    rare_counts = splits.loc[splits["tasting_category_fine"] == rare_style].groupby(
        ["repeat", "fold"]
    ).size()
    if len(rare_counts) != N_REPEATS * N_SPLITS or not rare_counts.eq(1).all():
        raise RuntimeError(f"rare style {rare_style!r} is not represented once per fold")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    splits.to_csv(RESULT_DIR / "q1_cv_splits.csv", index=False)
    manifest = {
        "method": "RepeatedStratifiedKFold",
        "stratification": "tasting_category_fine",
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "random_state": RANDOM_STATE,
        "n_rows": len(splits),
        "validation_rows_per_repeat": len(x),
        "rare_style": rare_style,
        "rare_style_count": int((style == rare_style).sum()),
    }
    (RESULT_DIR / "q1_cv_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

