"""Train new target specialists; select on valid prefix, audit on its suffix.

All input features come from X. No lagged labels are input features. The
validation suffix has been examined by earlier experiments, so it is a
robustness diagnostic, not a fresh untouched holdout.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config, resolve_data_path
from src.dataset import (load_panel, drop_other_label, slice_split, flatten_masked_rows,
                         resolve_cat_indices, cat_feature_col_indices, feature_names)
from src.models.baseline import LightGBMBaseline
from src.models.fusion import panel_cs_rank
from src.metrics import rank_ic_series
from src.submit import save_submission

OUT = ROOT / 'outputs'


def make_targets(split, coords):
    """Rank the full labeled universe before sampling, per day."""
    targets = {name: np.empty(len(coords), dtype=np.float32)
               for name in ('rank', 'industry_residual')}
    day_edges = np.r_[0, np.flatnonzero(np.diff(coords[:, 0])) + 1, len(coords)]
    for left, right in zip(day_edges[:-1], day_edges[1:]):
        t = int(coords[left, 0])
        y = split['y1'][t]
        mask = np.asarray(split['mask_y'][t], bool) & np.isfinite(y)
        z = np.zeros(len(y), np.float32)
        ranks = rankdata(y[mask], method='average')
        z[mask] = ((ranks - ranks.mean()) / max(ranks.std(), 1e-8)).astype(np.float32)
        residual = z.copy()
        groups = split['cat_x'][t, :, 6]
        for group in np.unique(groups[mask]):
            members = mask & (groups == group)
            if members.sum() >= 5:
                residual[members] -= z[members].mean()
        stocks = coords[left:right, 1]
        targets['rank'][left:right] = z[stocks]
        targets['industry_residual'][left:right] = residual[stocks]
    return targets


def stats(s):
    return dict(full=float(s.mean()), prefix=float(s[:120].mean()),
                suffix=float(s[120:].mean()), last60=float(s[-60:].mean()),
                blocks=[float(a.mean()) for a in np.array_split(s, 4)])


def main():
    started = time.time()
    cfg = load_config('configs/hist_lgbm_alpha_x.yaml')
    features = dict(cfg['features'])
    features['max_train_stocks_per_day'] = 400
    cats = resolve_cat_indices(features)
    print('LOADING one panel; 400 stocks/day, full train, three target variants', flush=True)
    data = load_panel(str(resolve_data_path(cfg)))
    drop_other_label(data, 'y1')
    train, valid, test = [slice_split(data, split) for split in ('train','valid','test')]
    print('BUILD FEATURES', flush=True)
    x, raw_y, coords = flatten_masked_rows(train, cat_indices=cats, feature_cfg=features,
                                           require_label=True, seed=42)
    assert np.isfinite(raw_y).all()
    print('FEATURES', x.shape, 'elapsed', round(time.time()-started), flush=True)
    targets = make_targets(train, coords)
    targets['raw_control'] = raw_y
    cat_cols = cat_feature_col_indices(99, cats, features)
    names = feature_names(99, cats, features)
    params = dict(objective='regression', metric='l2', learning_rate=.05,
                  num_leaves=31, min_data_in_leaf=100, lambda_l2=5.0,
                  feature_fraction=.8, num_threads=6, verbosity=-1,
                  seed=42, deterministic=True, force_col_wise=True, max_bin=127)
    ds = lgb.Dataset(x, label=raw_y, categorical_feature=cat_cols,
                     feature_name=names, params=params, free_raw_data=True)
    ds.construct()
    del x, coords
    gc.collect()
    base = np.load(OUT/'fusion_alpha_platform_mix_valid.npy')
    base_rank = panel_cs_rank(base, valid['mask_x'])
    base_s = rank_ic_series(base, valid['y1'], valid['mask_y'])
    report = dict(baseline=stats(base_s), models={}, selection_days=[0,120],
                  audit_days=[120,243], inputs='X only', train_stocks_per_day=400,
                  params=params, rounds=450)
    preds = {}
    for kind in ('raw_control', 'rank', 'industry_residual'):
        ds.set_label(targets[kind])
        print('TRAINING', kind, flush=True)
        def progress(env):
            if (env.iteration+1) % 100 == 0:
                print(kind, 'iteration', env.iteration+1, 'elapsed', round(time.time()-started), flush=True)
        booster = lgb.train(params, ds, num_boost_round=450, callbacks=[progress])
        model = LightGBMBaseline(params, feature_cfg=features, seed=42)
        model.booster = booster
        model.feature_name_list = names
        model.cat_feature_indices = cat_cols
        model.save(ROOT/f'checkpoints/task1_rank_specialist_{kind}.txt')
        print('PREDICT VALID', kind, flush=True)
        pv = model.predict_panel(valid)
        assert np.isfinite(pv).all()
        np.save(OUT/f'task1_rank_specialist_{kind}_valid.npy', pv)
        preds[kind] = pv
        s = rank_ic_series(pv, valid['y1'], valid['mask_y'])
        report['models'][kind] = stats(s)
        print('SINGLE', kind, json.dumps(stats(s)), flush=True)
        del model, booster
        gc.collect()
    del ds, targets, raw_y
    gc.collect()
    # Prespecified coarse candidates, plus no change; choose only on first 120 days.
    candidates = []
    for name, pv in preds.items():
        if name == 'raw_control':
            continue
        pr = panel_cs_rank(pv, valid['mask_x'])
        for weight in (.15, .30, .50):
            blend = (1-weight)*base_rank + weight*pr
            s = rank_ic_series(blend, valid['y1'], valid['mask_y'])
            candidates.append((s[:120].mean(), name, weight, blend, s))
    best = max(candidates, key=lambda c:c[0])
    _, kind, weight, pv, s = best
    delta = s-base_s
    accepted = (delta[:120].mean()>0.0003 and delta[120:].mean()>0.0003
                and delta[-60:].mean()>0 and delta.mean()>0.0005)
    report['coarse_candidates'] = [dict(model=c[1],weight=c[2],**stats(c[4])) for c in candidates]
    report['selected'] = dict(model=kind, weight=weight, **stats(s),
                              delta=stats(delta), accepted=bool(accepted))
    if accepted:
        print('PASSED SUFFIX AUDIT; PREDICT TEST', kind, flush=True)
        model = LightGBMBaseline(params, feature_cfg=features, seed=42)
        model.load(ROOT/f'checkpoints/task1_rank_specialist_{kind}.txt')
        pt = model.predict_panel(test)
        np.save(OUT/f'task1_rank_specialist_{kind}_test.npy',pt)
        base_t = np.load(OUT/'fusion_alpha_platform_mix_test.npy')
        pt = ((1-weight)*panel_cs_rank(base_t,test['mask_x'])
              +weight*panel_cs_rank(pt,test['mask_x']))
        tag=f'task1_fusion_rank_specialist_{kind}'
        np.save(OUT/f'{tag}_valid.npy',pv)
        np.save(OUT/f'{tag}_test.npy',pt)
        save_submission(pt, ROOT/f'submissions/{tag}.npy')
        report['submission']=f'submissions/{tag}.npy'
    report['seconds']=time.time()-started
    (OUT/'task1_rank_specialists_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RESULT',json.dumps(report['selected']), flush=True)
    print('DONE seconds',round(time.time()-started),flush=True)


if __name__ == '__main__':
    main()
