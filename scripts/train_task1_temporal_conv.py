"""Train a new causal temporal architecture with train-internal epoch selection.

Official validation is a reused diagnostic, not an untouched holdout. Do not
choose epochs or small blend weights from the platform leaderboard.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config, resolve_data_path
from src.dataset import load_panel, drop_other_label
from src.metrics import rank_ic_series
from src.models.temporal_conv import TemporalConvModel
from src.models.fusion import panel_cs_rank
from src.submit import save_submission

OUT = ROOT/'outputs'
STEM = 'task1_tcn_multiscale_v1'


def light_split(data, start, end):
    return dict(start=start, end=end, **{k: data[k][start:end]
                for k in ('y1','mask_x','mask_y')})


def stats(series):
    if not np.isfinite(series).all():
        raise ValueError('Invalid evaluation day; refuse to silently skip it')
    return dict(full=float(series.mean()), suffix=float(series[120:].mean()),
                last60=float(series[-60:].mean()),
                quarters=[float(s.mean()) for s in np.array_split(series, 4)])


def main():
    started = time.time()
    torch.set_num_threads(4)
    cfg = load_config('configs/gru_x6_with_today.yaml')
    model_cfg = dict(num_indices=cfg['features']['num_indices'], length=32,
                     include_current_day=True, source='cs_zscore', clip=5.,
                     max_train_stocks_per_day=1000, hidden_size=24,
                     head_dropout=.1, loss='mse', lr=.0005, weight_decay=.001,
                     max_epochs=5, patience=2, log_every_days=100, rnn='tcn')
    print('LOAD; X-only TCN, 32-day window, dilations 1/3/9', flush=True)
    data = load_panel(str(resolve_data_path(cfg)))
    drop_other_label(data,'y1')
    va, te = int(data['valid_start_idx']), int(data['test_start_idx'])
    start = max(int(data['train_start_idx']), va-800)
    hold_start, gap = va-120, 20
    hold = light_split(data,hold_start,va)
    valid = light_split(data,va,te)
    test = light_split(data,te,len(data['mask_x']))
    model = TemporalConvModel(model_cfg,seed=2026)
    model.prepare_features(data)
    for key in ('num_x','cat_x'):
        data.pop(key,None)
    gc.collect()
    print('SELECT EPOCHS',start,hold_start-gap,'hold',hold_start,va,flush=True)
    selection = model.fit(data,start,hold_start-gap,valid=hold)
    best_epoch = max(selection['history'],key=lambda row:row['valid_ic'])['epoch']
    print('LOCKED EPOCHS',best_epoch,'REFIT full train excluding last 20 days',flush=True)
    model.cfg['max_epochs'] = best_epoch
    final_fit = model.fit(data,start,va-gap,valid=None)
    model.save(ROOT/f'checkpoints/{STEM}.pt')
    pv = model.predict_panel(valid,data)
    np.save(OUT/f'{STEM}_valid.npy',pv)
    base = np.load(OUT/'fusion_alpha_platform_mix_valid.npy')
    best = np.load(OUT/'task1_fusion_trendmix_x2_w10_valid.npy')
    score = lambda p: rank_ic_series(p,valid['y1'],valid['mask_y'])
    base_s = score(base)
    rank = panel_cs_rank(pv,valid['mask_x'])
    # Freeze 25% weight before viewing official validation; no weight grid.
    blend = .75*panel_cs_rank(base,valid['mask_x'])+.25*rank
    submission_valid = .75*panel_cs_rank(best,valid['mask_x'])+.25*rank
    delta = score(blend)-base_s
    final_delta = score(submission_valid)-score(best)
    passed = (delta.mean()>.0005 and delta[120:].mean()>.0005
              and delta[-60:].mean()>0 and
              sum(x.mean()>0 for x in np.array_split(delta,4))>=3
              and final_delta.mean()>.0005)
    report = dict(config=model_cfg, seed=2026, train_start=start, train_end=va-gap,
                  selection_end=hold_start-gap, hold_start=hold_start, hold_end=va,
                  gap_days=gap, selected_epochs=best_epoch, selection=selection,
                  fit=final_fit, single=stats(score(pv)), x_baseline=stats(base_s),
                  x_blend=stats(score(blend)), delta=stats(delta),
                  current_best=stats(score(best)), candidate=stats(score(submission_valid)),
                  candidate_delta=stats(final_delta), accepted=bool(passed))
    print('AUDIT',json.dumps(report,ensure_ascii=False),flush=True)
    if passed:
        pt = model.predict_panel(test,data)
        np.save(OUT/f'{STEM}_test.npy',pt)
        best_t = np.load(OUT/'task1_fusion_trendmix_x2_w10_test.npy')
        candidate = .75*panel_cs_rank(best_t,test['mask_x'])+.25*panel_cs_rank(pt,test['mask_x'])
        if not np.isfinite(candidate).all() or candidate.shape != test['mask_x'].shape:
            raise ValueError('Invalid test predictions')
        tag = 'task1_fusion_tcn_multiscale_v1'
        np.save(OUT/f'{tag}_valid.npy',submission_valid)
        np.save(OUT/f'{tag}_test.npy',candidate)
        path = save_submission(candidate,ROOT/f'submissions/{tag}.npy')
        report['submission']=str(path)
    report['seconds']=time.time()-started
    (OUT/f'{STEM}_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('DONE accepted=',passed,'seconds=',round(report['seconds']),flush=True)


if __name__ == '__main__':
    main()
