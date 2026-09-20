from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_config,resolve_data_path
from src.dataset import load_panel,drop_other_label,slice_split
from src.metrics import rank_ic_series
from src.submit import save_submission
OUT=ROOT/'outputs'
def pct(x,m):
    x=np.asarray(x,np.float32); m=np.asarray(m,bool)&np.isfinite(x); z=np.zeros(x.shape,np.float32)
    if m.sum()>=2:
        r=rankdata(x[m],method='average').astype(np.float32); z[m]=(r-1)/max(r.size-1,1)
    return z
def run(cur,mask,p1,p2,n,beta):
    out=np.zeros_like(cur,np.float32); a=np.array(p1,np.float32); b=np.array(p2,np.float32); va=np.isfinite(a); vb=np.isfinite(b)
    for t in range(cur.shape[0]):
        m=np.asarray(mask[t],bool); c=pct(cur[t],m); r1=pct(a,m&va); r2=pct(b,m&vb); use=m&va&np.isfinite(cur[t]); out[t]=c
        if t<n: out[t,use]=np.clip(r1+beta*(r1-r2),0,1)[use]
        b,vb=a,va; a,va=out[t],m&np.isfinite(cur[t])
    return out
cfg=load_config('configs/hist_lgbm_alpha_x.yaml'); d=load_panel(str(resolve_data_path(cfg,None))); drop_other_label(d,'y1'); v,t=slice_split(d,'valid'),slice_split(d,'test'); y=np.asarray(d['y1'],np.float32); cv=np.load(OUT/'fusion_alpha_platform_mix_valid.npy'); ct=np.load(OUT/'fusion_alpha_platform_mix_test.npy'); p1v,p2v=y[v['start']-1],y[v['start']-2]; p1t,p2t=y[t['start']-1],y[t['start']-2]; pv=run(cv,v['mask_x'],p1v,p2v,2,.5); pt=run(ct,t['mask_x'],p1t,p2t,2,.5); tag='task1_fusion_holdtrend_b050_n02'; np.save(OUT/f'{tag}_valid.npy',pv); np.save(OUT/f'{tag}_test.npy',pt); save_submission(pt,ROOT/f'submissions/{tag}.npy'); print(tag,float(np.nanmean(rank_ic_series(pv,v['y1'],v['mask_y']))))

