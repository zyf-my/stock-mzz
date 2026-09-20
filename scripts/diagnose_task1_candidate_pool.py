import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_split_cache
from src.metrics import rank_ic_series

v = load_split_cache('valid')
base = np.load(Path(__file__).resolve().parents[1] / 'outputs/task1_fusion_trendmix_x2_w10_valid.npy')
bs = rank_ic_series(base, v['y1'], v['mask_y'])
rows = []
for p in sorted((Path(__file__).resolve().parents[1] / 'outputs').glob('task1_fusion*_valid.npy')):
    if p.name == 'task1_fusion_trendmix_x2_w10_valid.npy':
        continue
    try:
        a = np.load(p)
        s = rank_ic_series(a, v['y1'], v['mask_y'])
    except Exception:
        continue
    cs = []
    for t, m in enumerate(v['mask_x']):
        z = m & np.isfinite(base[t]) & np.isfinite(a[t])
        if z.sum() > 10:
            cs.append(spearmanr(base[t, z], a[t, z]).statistic)
    if cs:
        rows.append(dict(full=float(s.mean()), suffix=float(s[120:].mean()),
                         last60=float(s[-60:].mean()), corr=float(np.mean(cs)),
                         name=p.stem))
rows.sort(key=lambda x: x['full'], reverse=True)
print('BASE', float(bs.mean()), float(bs[120:].mean()), float(bs[-60:].mean()))
print('TOP_FULL')
for x in rows[:35]: print(x)
print('TOP_SUFFIX')
for x in sorted(rows, key=lambda x: (x['suffix'], x['corr']), reverse=True)[:35]: print(x)
