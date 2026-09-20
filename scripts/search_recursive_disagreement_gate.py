"""Search a causal disagreement gate for the two-day y1 trend signal."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.config import load_config,resolve_data_path
from src.dataset import load_panel,drop_other_label,slice_split
from src.metrics import rank_ic_series
from src.models.fusion import panel_cs_rank
from src.submit import save_submission
OUT=ROOT/'outputs'

def pct(x,m):
    x=np.asarray(x,dtype=np.float32); m=np.asarray(m,bool)&np.isfinite(x); z=np.zeros(x.shape,np.float32)
    if m.sum()>=2:
        r=rankdata(x[m],method='average').astype(np.float32); z[m]=(r-1)/max(r.size-1,1)
    return z

def run(cur,mask,p1,p2,w_agree,w_disagree,q,decay,scale,beta=1.0):
    out=np.zeros_like(cur,dtype=np.float32); a=np.asarray(p1,dtype=np.float32).copy(); b=np.asarray(p2,dtype=np.float32).copy(); va=np.isfinite(a); vb=np.isfinite(b)
    for t in range(cur.shape[0]):
        m=np.asarray(mask[t],bool); c=pct(cur[t],m); r1=pct(a,m&va); r2=pct(b,m&vb); use=m&va&np.isfinite(cur[t])
        sig=np.clip(r1+beta*(r1-r2),0,1); dis=np.abs(c-r1); wt0=float(decay)**(t/float(scale)); gate=np.where(dis<=q,w_agree,w_disagree)*wt0
        out[t]=c; out[t,use]=((1-gate[use])*c[use]+gate[use]*sig[use]).astype(np.float32)
        b,vb=a,va; a,va=out[t],m&np.isfinite(cur[t])
    return out

def main():
    cfg=load_config('configs/hist_lgbm_alpha_x.yaml'); d=load_panel(str(resolve_data_path(cfg,None))); drop_other_label(d,'y1'); v,t=slice_split(d,'valid'),slice_split(d,'test'); y=np.asarray(d['y1'],np.float32)
    cv=np.load(OUT/'fusion_alpha_platform_mix_valid.npy'); ct=np.load(OUT/'fusion_alpha_platform_mix_test.npy'); base=rank_ic_series(panel_cs_rank(cv,v['mask_x']),v['y1'],v['mask_y'])
    p1v,p2v=y[v['start']-1],y[v['start']-2]; p1t,p2t=y[t['start']-1],y[t['start']-2]; rows=[]; vals=[]
    for q in [.15,.25,.35,.45,.55]:
      for wa in [.6,.8,1.0]:
       for wd in [0.0,.2,.4]:
        for decay in [.002,.005,.01]:
         pv=run(cv,v['mask_x'],p1v,p2v,wa,wd,q,decay,45); s=rank_ic_series(pv,v['y1'],v['mask_y']); row=(float(np.nanmean(s)),float(np.nanmean(s[:120])),float(np.nanmean(s[120:])),q,wa,wd,decay); rows.append(row); vals.append((row,pv))
    rows.sort(reverse=True,key=lambda x:x[0]); print('BEST',rows[:15]); (OUT/'recursive_gate_grid.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for row,pv in vals:
        if row!=rows[0]: continue
        _,_,_,q,wa,wd,decay=row; pt=run(ct,t['mask_x'],p1t,p2t,wa,wd,q,decay,45); tag=f'recursive_gate_q{int(q*100):02d}_a{int(wa*100):03d}_d{int(wd*100):03d}_r{int(decay*1000):03d}'; np.save(OUT/f'fusion_alpha_{tag}_valid.npy',pv); np.save(OUT/f'fusion_alpha_{tag}_test.npy',pt); save_submission(pt,ROOT/f'submissions/task1_fusion_alpha_{tag}.npy'); print('WROTE',tag)
        break

if __name__=='__main__': main()
