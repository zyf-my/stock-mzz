"""Causal filtering of industry/stock prediction components; no label inputs.

Four fixed hypotheses are selected on the first 120 validation days and then
audited on the remaining days (reused diagnostics, not an untouched holdout).
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
BEST = 'task1_fusion_trendmix_x2_w10'
BASE = 'fusion_alpha_platform_mix'
MODES = ('all_smooth', 'stock_smooth', 'industry_smooth', 'stock_trend')


def normalized_rank(p, mask):
    out = np.zeros_like(p, dtype=np.float32)
    for t, m in enumerate(np.asarray(mask, bool)):
        if m.sum() >= 2:
            r = rankdata(p[t, m])
            out[t, m] = (r - r.mean()) / max(r.std(), 1e-8)
    return out


def components(r, mask, industry):
    sector = np.zeros_like(r)
    for t, m in enumerate(np.asarray(mask, bool)):
        for g in np.unique(industry[t, m]):
            ix = m & (industry[t] == g)
            if ix.sum() >= 5:
                sector[t, ix] = r[t, ix].mean()
    return sector, r-sector


def ema(p, mask, industry=None):
    out = p.copy()
    for t in range(1, len(p)):
        use = mask[t] & mask[t-1]
        if industry is not None:
            use &= industry[t] == industry[t-1]
        out[t, use] = .5*p[t, use] + .5*out[t-1, use]
    return out


def candidates(base, best, mask, industry):
    r = normalized_rank(base, mask)
    anchor = normalized_rank(best, mask)
    sector, stock = components(r, mask, industry)
    stock_smooth = ema(stock, mask, industry)
    changes = dict(all_smooth=ema(r, mask)-r,
                   stock_smooth=stock_smooth-stock,
                   industry_smooth=ema(sector, mask, industry)-sector,
                   stock_trend=stock-stock_smooth)
    # Fixed 50% correction, not a weight sweep.
    return {k: np.where(mask, anchor+.5*v, 0).astype(np.float32)
            for k, v in changes.items()}


def stats(s):
    if not np.isfinite(s).all():
        raise ValueError('Nonfinite daily RankIC')
    return dict(full=float(s.mean()), prefix=float(s[:120].mean()),
                suffix=float(s[120:].mean()), last60=float(s[-60:].mean()),
                blocks=[float(x.mean()) for x in np.array_split(s, 4)])


def main():
    v = load_split_cache('valid')
    mask = np.asarray(v['mask_x'], bool)
    base = np.load(OUT/f'{BASE}_valid.npy')
    best = np.load(OUT/f'{BEST}_valid.npy')
    score = lambda x: rank_ic_series(x, v['y1'], v['mask_y'])
    original = score(best)
    variants = candidates(base, best, mask, v['industry'])
    # Verify causal prefix invariance on actual inputs before considering scores.
    prefix = candidates(base[:120], best[:120], mask[:120], v['industry'][:120])
    for k in MODES:
        np.testing.assert_array_equal(prefix[k], variants[k][:120])
    rows = {}
    for k, p in variants.items():
        s = score(p)
        rows[k] = dict(score=stats(s), delta=stats(s-original))
    winner = max(MODES, key=lambda k: rows[k]['delta']['prefix'])
    delta = score(variants[winner])-original
    passed = (delta[:120].mean() > .0005 and delta[120:].mean() > .0005
              and delta[-60:].mean() > 0
              and sum(x.mean() > 0 for x in np.array_split(delta, 4)) >= 3)
    report = dict(baseline=stats(original), variants=rows, selected=winner,
                  accepted=bool(passed), causal_prefix_check=True,
                  caveat='Validation has been reused; platform score unknown')
    if passed:
        t = load_split_cache('test')
        combined_base = np.concatenate([base, np.load(OUT/f'{BASE}_test.npy')])
        combined_best = np.concatenate([best, np.load(OUT/f'{BEST}_test.npy')])
        combined_mask = np.concatenate([mask, np.asarray(t['mask_x'], bool)])
        combined_industry = np.concatenate([v['industry'], t['industry']])
        # Carry only prediction state across the validation/test boundary.
        p = candidates(combined_base, combined_best, combined_mask, combined_industry)[winner][len(mask):]
        assert p.shape == (442, 5282) and p.dtype == np.float32 and np.isfinite(p).all()
        tag = f'task1_fusion_temporal_{winner}'
        np.save(OUT/f'{tag}_valid.npy', variants[winner])
        np.save(OUT/f'{tag}_test.npy', p)
        report['submission'] = str(save_submission(p, ROOT/f'submissions/{tag}.npy'))
    (OUT/'task1_temporal_components_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
