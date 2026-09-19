"""Build y1 proxy panels for y2_ortho GRU (test uses task1 GRU pred, not y1 labels)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split  # noqa: E402
from src.models.gru_ts import GRUModel  # noqa: E402

OUT = ROOT / "outputs" / "task2"
Y1_CFG = "configs/gru_x6_with_today.yaml"
Y1_CKPT = ROOT / "checkpoints/gru_x6_with_today.pt"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Y1_CFG)
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")

    if Y1_CKPT.is_file():
        feat = dict(cfg.get("features") or {})
        model_cfg = dict(cfg.get("model") or {})
        train_cfg = dict(cfg.get("train") or {})
        gru_cfg = {**feat, **model_cfg, **train_cfg, "label_key": "y1"}
        model = GRUModel(gru_cfg, seed=int(cfg.get("seed", 42)))
        model.prepare_features(data)
        model.load(Y1_CKPT)
        src = "task1_gru_x6"
    else:
        print(f"WARN: {Y1_CKPT} missing — fallback to y1 labels (valid ok, test would leak)")
        src = "y1_label_fallback"
        model = None

    for name, split in [("valid", valid), ("test", test)]:
        if model is not None:
            pred = model.predict_panel(split, data)
        else:
            pred = np.asarray(split["y1"], dtype=np.float32)
        path = OUT / f"y1_proxy_{name}.npy"
        np.save(path, pred.astype(np.float32))
        print(f"wrote {path} source={src} shape={pred.shape}")

    print("done")


if __name__ == "__main__":
    main()
