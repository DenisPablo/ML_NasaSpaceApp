from common import *
from scipy import stats
h=con.sql(f"""SELECT satelite, anio, median(CAST(substr(hora_utc,1,2) AS INT)*60+CAST(substr(hora_utc,3,2) AS INT) - 180) AS min_local,
  count(*) n, median(frp_mw) frp_med, avg(CASE WHEN frp_mw<10 THEN 1 ELSE 0 END) frac_frp_lt10
  FROM read_parquet('{D}/focos/*/*.parquet', hive_partitioning=1) WHERE dia_noche='D' AND sensor='MODIS' GROUP BY 1,2 ORDER BY 1,2""").df()
h["hora_local"]=(h.min_local/60).round(2)
print(h.pivot(index="anio",columns="satelite",values="hora_local").to_string())
print(h.pivot(index="anio",columns="satelite",values="frac_frp_lt10").round(3).to_string())
# FRP small-fire share vs annual V/M
mm=pd.read_parquet("mm.parquet"); ya=mm.groupby("anio")[["V","M"]].sum(); ya["r"]=ya.V/ya.M
fr=con.sql(f"""SELECT anio, avg(CASE WHEN frp_mw<10 THEN 1 ELSE 0 END) f10, median(frp_mw) med FROM read_parquet('{D}/focos/*/*.parquet', hive_partitioning=1) WHERE sensor='MODIS' GROUP BY 1 ORDER BY 1""").df().set_index("anio")
j=ya.join(fr); print("Spearman V/M vs frac MODIS FRP<10:", stats.spearmanr(j.r,j.f10), " vs mediana FRP:", stats.spearmanr(j.r,j.med))
print("frac FRP<10 MODIS por año:", fr.f10.round(3).to_dict())
# composition effect: national implied factor with 2012-25 prov-month factors, weights = MODIS composition of each era
pub=pub_factors(); pub=pub[pub.tipo=="modis"]
pdd,op=load(); pdd["ts"]=pd.to_datetime(pdd.fecha); pdd["mes"]=pdd.ts.dt.month; pdd["M"]=pdd.n_terra+pdd.n_aqua
def implied(a,b):
    s=pdd[(pdd.ts>=a)&(pdd.ts<=b)].groupby(["provincia_id","mes"]).M.sum().reset_index().merge(pub,on=["provincia_id","mes"])
    return (s.M*s.factor).sum()/s.M.sum()
print("factor nacional implícito, composición 2003-2011:", round(implied("2003-01-01","2011-12-31"),3), " composición 2012-2025:", round(implied("2012-01-01","2025-12-31"),3))
# time-distance extrapolation: fit on one year, predict another, national annual abs error vs distance
res=[]
yrs=sorted(mm.anio.unique())
for a in yrs:
    f=CAL.fit_hierarchical(mm[mm.anio==a])
    for b in yrs:
        if a==b: continue
        te=mm[mm.anio==b]; p=CAL.predict(te,f).sum(); res.append((abs(a-b),abs(p-te.V.sum())/te.V.sum(), np.abs(CAL.predict(te,f)-te.V).sum()/te.V.sum()))
r=pd.DataFrame(res,columns=["d","err_nat","err_pm"])
t=r.groupby("d")[["err_nat","err_pm"]].mean().mul(100).round(1); print(t.T.to_string())
print("Spearman distancia vs error nac:", stats.spearmanr(r.d,r.err_nat), " Theil slope pp/año:", round(100*stats.theilslopes(r.err_nat,r.d)[0],3))
