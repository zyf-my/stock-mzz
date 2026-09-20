"""Search small, predeclared temporal corrections around the locked best.

The platform rejected the strong all-stock smoother. This script measures
smaller corrections and slower EMA variants, with a suffix and block audit.
It never uses test labels and only emits a candidate if the correction is
positive in both validation halves.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import load_split_cache
from src.metrics import rank_ic_series
from src.submit import save_submission

OUT = ROOT / 'outputs'
BASE = 'fusion_alpha_platform_mix'
BEST = 'task1_fusion_trendmix_x2_w10'


def rank_panel(p, mask):
    out = np.zeros_like(p, dtype=np.float32)
    for t, m in enumerate(np.asarray(mask, bool)):
        if m.sum() >= 2:
            z = rankdata(p[t, m]).astype(np.float32)
            out[t, m] = (z - z.mean()) / max(z.std(), 1e-8)
    return out


def ema(p, mask, alpha):
    out = p.copy()
    for t in range(1, len(p)):
        use = mask[t] & mask[t - 1]
        out[t, use] = alpha * p[t, use] + (1 - alpha) * out[t - 1, use]
    return out


def stats(s):
    return dict(full=float(s.mean()), prefix=float(s[:120].mean()),
                suffix=float(s[120:].mean()), last60=float(s[-60:].mean()),
                blocks=[float(x.mean()) for x in np.array_split(s, 4)])


def main():
    v = load_split_cache('valid')
    m = np.asarray(v['mask_x'], bool)
    base = np.load(OUT / f'{BASE}_valid.npy')
    best = np.load(OUT / f'{BEST}_valid.npy')
    rb, ra = rank_panel(base, m), rank_panel(best, m)
    score = lambda p: rank_ic_series(p, v['y1'], v['mask_y'])
    sb = score(best)
    rows = []
    for alpha in (0.25, 0.50, 0.75):
        smooth = ema(rb, m, alpha)
        correction = smooth - rb
        for weight in (0.10, 0.20, 0.30):
            p = np.where(m, ra + weight * correction, 0).astype(np.float32)
            s = score(p)
            d = s - sb
            rows.append(dict(alpha=alpha, weight=weight, score=stats(s), delta=stats(d), pred=p))
    # Use the most conservative candidate among those positive in both halves.
    eligible = [r for r in rows if r['delta']['prefix'] > 0 and r['delta']['suffix'] > 0
                and r['delta']['last60'] >= 0 and sum(x > 0 for x in r['delta']['blocks']) >= 3]
    chosen = max(eligible, key=lambda r: (r['delta']['suffix'], r['delta']['full'])) if eligible else None
    report = dict(baseline=stats(sb), candidates=[{k:v for k,v in r.items() if k != 'pred'} for r in rows],
                  chosen=None if chosen is None else {k:v for k,v in chosen.items() if k != 'pred'},
                  accepted=bool(chosen is not None), caveat='Platform score unknown')
    if chosen is not None:
        t = load_split_cache('test')
        mt = np.asarray(t['mask_x'], bool)
        base_all = np.concatenate([base, np.load(OUT / f'{BASE}_test.npy')])
        best_all = np.concatenate([best, np.load(OUT / f'{BEST}_test.npy')])
        mask_all = np.concatenate([m, mt])
        corr_all = ema(rank_panel(base_all, mask_all), mask_all, chosen['alpha']) - rank_panel(base_all, mask_all)
        pred = np.where(mt, rank_panel(best_all, mask_all)[len(m):] + chosen['weight'] * corr_all[len(m):], 0).astype(np.float32)
        tag = f"task1_fusion_temporal_conservative_a{int(chosen['alpha']*100):02d}_w{int(chosen['weight']*100):02d}"
        np.save(OUT / f'{tag}_valid.npy', chosen['pred'])
        np.save(OUT / f'{tag}_test.npy', pred)
        report['submission'] = str(save_submission(pred, ROOT / f'submissions/{tag}.npy'))
    (OUT / 'task1_conservative_temporal_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
