from common import *
mm = pd.read_parquet("mm.parquet")
rg = rangos()
def loyo(m, k, kp=None):
    out=[]
    for y in sorted(m.anio.unique()):
        tr, te = m[m.anio!=y], m[m.anio==y].copy()
        f = CAL.fit_hierarchical(tr, k)
        if kp is not None:  # provincial level K different
            import numpy as _n
        te["pred"]=CAL.predict(te,f); out.append(te)
    return pd.concat(out)
def err(d):
    v=d.V.sum(); return 100*np.abs(d.pred-d.V).sum()/v
print("== K por LOYO (error mensual prov. ponderado, %; error anual nacional medio, %)")
res={}
for k in [0.001,25,50,100,200,400,800,1600,5000,1e9]:
    d=loyo(mm,k); a=d.groupby("anio")[["V","pred"]].sum()
    res[k]=d
    print(f"K={k:>10}: err_prov_mes={err(d):6.2f}  err_anual_nac={100*np.mean(np.abs(a.pred-a.V)/a.V):5.2f}  max_anual={100*np.max(np.abs(a.pred-a.V)/a.V):5.2f}")
d=res[200]
print("\n== Errores anuales nacionales LOYO K=200 (%)")
a=d.groupby("anio")[["V","pred"]].sum(); print((100*(a.pred-a.V)/a.V).round(1).to_dict())
# --- coverage LOYO by bin/province/year using published ranges
ok,lo,hi = cover(d.M.values,d.V.values,d.pred.values,rg); d["ok"]=ok
bins=pd.cut(d.M,[-1,0,5,20,100,500,1e12],labels=["0","1-5","6-20","21-100","101-500",">500"])
print("\n== Cobertura LOYO (rangos publicados) global:", round(100*ok.mean(),1))
print(d.groupby(bins,observed=True).ok.agg(["mean","size"]).assign(mean=lambda x:(100*x["mean"]).round(1)).to_string())
pc=d.groupby("provincia_id").ok.agg(["mean","size"]); pc["mean"]=(100*pc["mean"]).round(1)
print("por provincia:", pc["mean"].to_dict())
print("provincias con cobertura <70%:", (pc["mean"]<70).sum(), " >90%:", (pc["mean"]>90).sum())
# binomial test for each province vs 0.8
from scipy.stats import binomtest
pv={p:binomtest(int(round(r["mean"]/100*r["size"])),int(r["size"]),0.8).pvalue for p,r in pc.iterrows()}
print("provincias con p<0.05 (binomial vs 80%):", sorted([p for p,v in pv.items() if v<0.05]), " Bonferroni:", sorted([p for p,v in pv.items() if v<0.05/len(pv)]))
yc=d.groupby("anio").ok.mean().mul(100).round(1); print("por año:", yc.to_dict())
# Lower/upper miss balance
d["below"]=d.V<lo; d["above"]=d.V>hi
print("fuera por debajo %:", round(100*d.below.mean(),1), " por encima %:", round(100*d.above.mean(),1))
# bin 0: months with M=0 where V>9
z=d[d.M==0]; print("M=0: n",len(z)," V>9:",(z.V>9).sum()," max V", z.V.max(), " V==0 share", round((z.V==0).mean(),2))
# --- nationally aggregated range (sum of prov limits) coverage: monthly nat
d["lo"],d["hi"]=lo,hi
n=d.groupby(["anio","mes"])[["V","pred","lo","hi"]].sum()
print("Cobertura del rango NACIONAL mensual (suma de límites):", round(100*((n.V>=n.lo)&(n.V<=n.hi)).mean(),1), "% ; ancho relativo mediano", round(((n.hi-n.lo)/n.pred).median(),2))
# --- Forward test with ranges recalibrated on 2012-2019 only
tr=mm[mm.anio<=2019]; te=mm[mm.anio>=2020].copy()
rgf=CAL.calibrate_ranges(tr).drop(columns="n_meses"); f=CAL.fit_hierarchical(tr); te["pred"]=CAL.predict(te,f)
okf,_,_=cover(te.M.values,te.V.values,te.pred.values,rgf)
print("Cobertura hacia adelante (rangos y factores 2012-19 → 2020-25):", round(100*okf.mean(),1))
print(pd.Series(okf).groupby(pd.cut(te.M.values,[-1,0,5,20,100,500,1e12]).astype(str)).mean().mul(100).round(1).to_dict())
d.to_parquet("loyo200.parquet")
# --- LOPO: leave-one-province-out (factor = region prior from other provinces)
rows=[]
for p in sorted(mm.provincia_id.unique()):
    if p in C.LOW_DATA_PROVINCES: continue
    tr=mm[mm.provincia_id!=p]; te=mm[mm.provincia_id==p].copy()
    f=CAL.fit_hierarchical(tr)
    reg=C.PROVINCE_REGION[p]; fr=f.region_m.xs(reg,level="region")
    te["pred_lopo"]=te.M*te.mes.map(fr)
    te2=d[d.provincia_id==p]
    a=te.V.sum()
    rows.append((p, round(100*(te.pred_lopo.sum()-a)/a,1), round(100*np.abs(te.pred_lopo-te.V).sum()/a,1), round(100*np.abs(te2.pred-te2.V).sum()/te2.V.sum(),1)))
t=pd.DataFrame(rows,columns=["prov","sesgo_LOPO_%","err_LOPO_%","err_LOYO_%"]); print(t.to_string())
print("mediana |sesgo| LOPO:", t["sesgo_LOPO_%"].abs().median(), " err LOPO mediano", t["err_LOPO_%"].median(), " vs LOYO", t["err_LOYO_%"].median())
