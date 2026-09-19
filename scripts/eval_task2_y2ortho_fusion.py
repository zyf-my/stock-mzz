"""A3: gru_x6_y2ortho in TREE_BAG shell vs baseline."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_task2_recipe import build, metrics  # noqa: E402

OUT = ROOT / "outputs" / "task2"
BASE = OUT / "recipes" / "TREE_BAG_091445.json"


def _recipe() -> dict:
    p = BASE if BASE.is_file() else OUT / "recipes" / "OLD_091354.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    base = _recipe()
    base_m = metrics(base)
    print(f"TREE_BAG base valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f}")

    recipe = copy.deepcopy(base)
    recipe["name"] = "y2ortho_x6"
    recipe["x6"] = "gru_x6_y2ortho"
    m = metrics(recipe)
    ok = m["valid_ic"] >= base_m["valid_ic"] - 0.0005 and m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6
    print(f"y2ortho   valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} gate={'PASS' if ok else 'FAIL'}")

    if ok and m["valid_ic"] > base_m["valid_ic"] + 1e-8:
        from src.submit import save_submission
        import numpy as np

        out_t = build(recipe, "test")
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_y2ortho.npy")
        np.save(OUT / "fusion_y2ortho_valid.npy", build(recipe, "valid"))
        meta = {**recipe, **m, "baseline_valid": base_m["valid_ic"], "platform_test_prev": 0.091445}
        (OUT / "fusion_y2ortho_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("wrote submissions/task2_fusion_y2ortho.npy")

    print("VALID_RANKIC", f"{m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
