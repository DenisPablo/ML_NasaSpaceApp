from common import *
rng=np.random.default_rng(42)
pdd, op = load()
fit=CAL.fit_hierarchical(pd.read_parquet("mm.parquet"))
F=fit.prov_m.set_index(["provincia_id","mes"]).factor
days=CAL.valid_days(op,("SNPP","Terra","Aqua"))
x=pdd[pdd.fecha.isin(days)].copy(); x["ts"]=pd.to_datetime(x.fecha)
x=x[(x.ts>="2012-02-01")&(x.ts<="2025-12-31")]; x["ym"]=x.ts.dt.to_period("M"); x["M"]=x.n_terra+x.n_aqua; x["V"]=x.n_snpp
x["f"]=[F.get((p,m),np.nan) for p,m in zip(x.provincia_id,x.ts.dt.month)]
# National-level masking simulation: for each month, mask k random valid days, estimate those days with M*f (prov), measure |err| of monthly total
nat=x.groupby(["ym","fecha"]).apply(lambda d: pd.Series({"V":d.V.sum(),"E":(d.M*d.f).sum()}),include_groups=False).reset_index()
res=[]
for ym,d in nat.groupby("ym"):
    n=len(d); tot=d.V.sum()
    if n<25 or tot<200: continue
    for frac in [0.05,0.1,0.15,0.2,0.3,0.5,1.0]:
        k=max(1,int(round(frac*n))) if frac<1 else n
        for _ in range(20):
            idx=rng.choice(n,k,replace=False); est=tot-d.V.values[idx].sum()+d.E.values[idx].sum()
            res.append((frac,abs(est-tot)/tot))
r=pd.DataFrame(res,columns=["frac","err"])
print("== Simulación nacional: error relativo del total mensual según fracción estimada (meses con V>=200)")
print(r.groupby("frac").err.describe(percentiles=[.5,.9]).loc[:,["count","50%","90%"]].mul([1,100,100]).round(2).to_string())
# Province-level, same
pv=x.groupby(["provincia_id","ym","fecha"])[["V","M","f"]].agg({"V":"sum","M":"sum","f":"first"}).reset_index(); pv["E"]=pv.M*pv.f
res=[]
for (p,ym),d in pv.groupby(["provincia_id","ym"]):
    tot=d.V.sum(); n=pd.Period(ym).days_in_month
    if tot<50: continue
    # pad zero days
    V=np.zeros(n); E=np.zeros(n); di=pd.to_datetime(d.fecha).dt.day.values-1; V[di]=d.V.values; E[di]=d.E.values
    for frac in [0.1,0.2,0.5]:
        k=int(round(frac*n))
        for _ in range(5):
            idx=rng.choice(n,k,replace=False); est=tot-V[idx].sum()+E[idx].sum(); res.append((frac,abs(est-tot)/tot))
r2=pd.DataFrame(res,columns=["frac","err"])
print("== Provincia-mes (V>=50):"); print(r2.groupby("frac").err.describe(percentiles=[.5,.9]).loc[:,["count","50%","90%"]].mul([1,100,100]).round(2).to_string())
# --- threshold 21: LOYO relative error by finer M bins (province-month, full month estimated)
d=pd.read_parquet("loyo200.parquet"); d=d[d.M>0]
b=pd.cut(d.M,[0,5,10,15,20,25,30,50,100,500,1e9])
t=d.groupby(b,observed=True).apply(lambda g: pd.Series({"n":len(g),"err_rel_med":100*np.median(np.abs(g.pred-g.V)/g.pred),"p90":100*np.quantile(np.abs(g.pred-g.V)/g.pred,.9),"ancho80":100*(np.quantile(g.V/g.pred,.9)-np.quantile(g.V/g.pred,.1))}),include_groups=False)
print("== Error relativo LOYO por M (prov-mes)"); print(t.round(1).to_string())
