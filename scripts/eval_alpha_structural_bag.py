"""Evaluate a fixed 50/50 structural bag against the frozen platform winner.

No weight search. Validation has been reused extensively, so time blocks are
diagnostics rather than an independent holdout or a generalization guarantee.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import load_eval_splits
from src.metrics import rank_ic_series
from src.models.fusion import linear_blend, panel_cs_rank
from src.submit import save_submission
from eval_alpha_disagree_gate import build_final, daily_disagreement

OUT = ROOT / 'outputs'


def load(stem, split):
    return np.load(OUT / f'{stem}_{split}.npy', allow_pickle=False)


def predict(split, mask, spec, new_stem):
    a = load('hist_lgbm_alpha_x_base', split)
    b = load('hist_lgbm_alpha_x_hard', split)
    d = daily_disagreement(a, b, mask)
    w = np.where(d >= spec['disagreement_cut'],
                 spec['ordinary_tree_weight_high_disagreement'],
                 spec['ordinary_tree_weight_other_days'])[:, None]
    old = (w * a + (1 - w) * b).astype(np.float32)
    baseline = load('baseline', split)
    branches = [load(stem, split) for stem in (
        'gru_x6_with_today', 'gru_only6_with_today', 'gru_next6_with_today',
        'cs_mlp', 'cs_mlp_only6')]
    other_rank = panel_cs_rank(load('fusion_x6_only6_mlp6', split), mask)

    def fuse(tree):
        alpha = build_final(linear_blend(tree, baseline, .7), *branches, mask)
        return linear_blend(panel_cs_rank(alpha, mask), other_rank, .82)

    restored = fuse(old)
    anchor = load('fusion_alpha_diverse_blend', split)
    # The persisted candidate was produced in a separate pass; ties can alter
    # a few float32 rank blends without changing the measured IC materially.
    return fuse(linear_blend(old, load(new_stem, split), .5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stem', default='hist_lgbm_alpha_x_keep40')
    args = parser.parse_args()
    splits, source = load_eval_splits(splits=('valid', 'test'), dump_if_missing=False)
    v = splits['valid']
    spec = json.loads((OUT / 'fusion_alpha_disagree_lock.json').read_text())
    pred = predict('valid', v['mask_x'], spec, args.stem)
    anchor = load('fusion_alpha_diverse_blend', 'valid')
    old = rank_ic_series(anchor, v['y1'], v['mask_y'])
    new = rank_ic_series(pred, v['y1'], v['mask_y'])
    cuts = [0, len(new)//4, len(new)//2, 3*len(new)//4, len(new)]
    delta = new - old
    block_deltas = [float(np.nanmean(delta[a:b])) for a, b in zip(cuts[:-1], cuts[1:])]
    hi = v['mask_x'].sum(axis=1) >= 4670
    gain = float(np.nanmean(delta))
    recent_gain = float(np.nanmean(delta[-60:]))
    # Fixed before inspecting this experiment; merely a conservative screen.
    eligible = gain >= .0002 and sum(x > 0 for x in block_deltas) >= 3 and recent_gain >= 0
    result = dict(stem=args.stem, cache_source=source, tree_new_weight=.5,
                  anchor_valid=float(np.nanmean(old)), valid=float(np.nanmean(new)),
                  valid_gain=gain, quarter_gains=block_deltas, last60_gain=recent_gain,
                  high_coverage_days=int(hi.sum()), high_coverage_ic=float(np.nanmean(new[hi])),
                  eligible=bool(eligible), platform_score=None,
                  caution='Reused validation, not an independent holdout.')
    if eligible:
        test_pred = predict('test', splits['test']['mask_x'], spec, args.stem)
        assert test_pred.shape == splits['test']['mask_x'].shape
        assert test_pred.dtype == np.float32 and np.isfinite(test_pred).all()
        tag = args.stem.removeprefix('hist_lgbm_') + '_bag50'
        np.save(OUT / f'fusion_{tag}_valid.npy', pred)
        np.save(OUT / f'fusion_{tag}_test.npy', test_pred)
        sub = ROOT / f'submissions/task1_fusion_{tag}.npy'
        save_submission(test_pred, sub)
        result['submission'] = str(sub)
    (OUT / f'{args.stem}_structural_bag_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
