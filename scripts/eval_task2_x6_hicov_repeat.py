"""Swap x6 branch to hicov_repeat GRU under OLD_091354 shell."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_task2_recipe import build, metrics  # noqa: E402

OUT = ROOT / "outputs" / "task2"


def main() -> None:
    recipe = json.loads((OUT / "recipes" / "OLD_091354.json").read_text(encoding="utf-8"))
    recipe = copy.deepcopy(recipe)
    recipe["name"] = "x6_hicov_repeat"
    recipe["x6"] = "gru_x6_hicov_repeat"
    recipe.pop("readonly", None)
    m = metrics(recipe)
    base = json.loads((OUT / "recipes" / "OLD_091354.json").read_text(encoding="utf-8"))
    from scripts.eval_task2_recipe import metrics as mfn

    base_m = mfn(base)
    print(f"x6_hicov_repeat valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f}")
    print(f"OLD              valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f}")
    ok = m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
    print(f"gate: {'PASS' if ok else 'FAIL'}")
    print("VALID_RANKIC", f"{m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
