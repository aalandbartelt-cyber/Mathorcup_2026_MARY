"""Losslessly compact E214 saved state after the formal numeric run.

Only the three registered target-style models are needed to replay E214.
Original state and checkpoint manifests are copied to a local audit archive;
generated samples, metrics, floor summaries and formal decision are untouched.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import shutil

from floor_aware_copula import FloorAwareStyleCopula
from run_e214_floor_aware_copula import CONFIG, RESULT, sha


TARGETS = ("Tripel", "Kriek", "Lambic")


def compact_state(original):
    original = dict(original)
    state = dict(original["model"])
    if not set(TARGETS).issubset(state["models"]):
        raise ValueError("registered target style missing from saved state")
    state["styles"] = [style for style in state["styles"] if style in TARGETS]
    state["models"] = {style: state["models"][style] for style in state["styles"]}
    original["model"] = state
    return original


def run(*, apply=False):
    archive = RESULT / "q3_state_precompact_E214"
    checkpoints = RESULT / "q3_checkpoints_E214"
    records = []
    for repeat in range(10):
        for fold in range(3):
            name = f"R{repeat:02d}F{fold}"
            slot = checkpoints / name
            state_path = slot / "model_state.pkl"
            manifest_path = slot / "manifest.json"
            old_state = state_path.read_bytes()
            if old_state[:2] == b"\x1f\x8b":
                raise ValueError(f"{name}: state already compressed; compaction is one-time")
            original = pickle.loads(old_state)
            compact = compact_state(original)
            model = FloorAwareStyleCopula.from_state(compact["model"])
            for style_index, style in enumerate(TARGETS):
                full_model = FloorAwareStyleCopula.from_state(original["model"])
                for seed_index, base_seed in enumerate(range(20261025, 20261030)):
                    seed = base_seed + repeat * 1000 + fold * 100 + style_index * 10
                    old = full_model.generate(style, 5, seed=seed)["values"]
                    new = model.generate(style, 5, seed=seed)["values"]
                    if not (old == new).all():
                        raise ValueError(f"{name}: compacted model changed generated values")
            packed = gzip.compress(pickle.dumps(compact, protocol=4), compresslevel=6, mtime=0)
            meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            if meta["files"]["model_state.pkl"] != sha(state_path):
                raise ValueError(f"{name}: original state hash changed")
            record = {"fold": name, "original_state_sha256": sha(state_path), "original_state_bytes": len(old_state),
                "original_manifest_sha256": sha(manifest_path), "compact_state_bytes": len(packed)}
            if apply:
                original_slot = archive / name
                original_slot.mkdir(parents=True, exist_ok=True)
                if any(original_slot.iterdir()):
                    raise ValueError(f"{name}: audit archive already exists")
                shutil.copy2(state_path, original_slot / "model_state.pkl")
                shutil.copy2(manifest_path, original_slot / "manifest.json")
                temporary = slot / "model_state.pkl.tmp"
                temporary.write_bytes(packed)
                temporary.replace(state_path)
                meta["files"]["model_state.pkl"] = sha(state_path)
                temporary_manifest = slot / "manifest.json.tmp"
                temporary_manifest.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temporary_manifest.replace(manifest_path)
                record.update(compact_state_sha256=sha(state_path), compact_manifest_sha256=sha(manifest_path))
            records.append(record)
    summary = {"experiment_id": "E214", "status": "complete" if apply else "dry_run", "n_folds": len(records),
        "target_styles_retained": list(TARGETS), "original_state_bytes": sum(x["original_state_bytes"] for x in records),
        "compact_state_bytes": sum(x["compact_state_bytes"] for x in records),
        "generated_and_metric_files_changed": False, "local_original_archive": str(archive), "folds": records}
    if apply:
        out = RESULT / "q3_state_compaction_E214.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "folds"}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(apply=args.apply)
