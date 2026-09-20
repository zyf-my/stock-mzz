"""Emit no-training blends from already evaluated Task 1 prediction panels."""
from pathlib import Path
import json
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import load_split_cache
from src.metrics import rank_ic_series
from src.models.fusion import panel_cs_rank
from src.submit import save_submission

OUT = ROOT / 'outputs'
BASE = 'task1_fusion_trendmix_x2_w10'
AUX = 'fusion_alpha_recursive_decay_w75_d50'


def stats(s):
    return dict(full=float(s.mean()), suffix=float(s[120:].mean()),
                last60=float(s[-60:].mean()),
                blocks=[float(x.mean()) for x in np.array_split(s, 4)])


def main():
    v, t = load_split_cache('valid'), load_split_cache('test')
    bv = panel_cs_rank(np.load(OUT / f'{BASE}_valid.npy'), v['mask_x'])
    av = panel_cs_rank(np.load(OUT / f'{AUX}_valid.npy'), v['mask_x'])
    bt = panel_cs_rank(np.load(OUT / f'{BASE}_test.npy'), t['mask_x'])
    at = panel_cs_rank(np.load(OUT / f'{AUX}_test.npy'), t['mask_x'])
    score = lambda p: rank_ic_series(p, v['y1'], v['mask_y'])
    report = {'base': stats(score(bv)), 'aux': stats(score(av)), 'variants': {}}
    for w in (0.05, 0.10, 0.15, 0.20):
        pv = ((1-w) * bv + w * av).astype(np.float32)
        pt = ((1-w) * bt + w * at).astype(np.float32)
        tag = f'task1_fusion_existing_recursive_w{int(w*100):02d}'
        np.save(OUT / f'{tag}_valid.npy', pv)
        np.save(OUT / f'{tag}_test.npy', pt)
        path = save_submission(pt, ROOT / f'submissions/{tag}.npy')
        s = score(pv)
        report['variants'][tag] = {'weight': w, **stats(s),
                                   'delta': stats(s-score(bv)), 'submission': str(path)}
    (OUT / 'task1_existing_blend_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
