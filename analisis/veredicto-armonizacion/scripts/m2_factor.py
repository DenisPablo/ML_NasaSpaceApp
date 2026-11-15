from common import *
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
mm = pd.read_parquet("mm.parquet")
# --- 1) National annual ratio, trend
ya = mm.groupby("anio")[["V","M"]].sum(); ya["r"]=ya.V/ya.M
print(ya.round(3).to_string())
r=ya.r; print("CV anual", round(r.std()/r.mean()*100,1), "% ; min-max", round(r.min(),2), round(r.max(),2))
ts = stats.theilslopes(r.values, r.index.values); print("Theil-Sen slope/yr", [round(x,4) for x in ts])
print("Spearman trend", stats.spearmanr(r.index, r.values))
# Poisson-ratio overdispersion: GLM V ~ offset(log M) + C(anio) vs common
g0 = smf.glm("V ~ 1", data=ya.reset_index(), family=sm.families.Poisson(), offset=np.log(ya.M.values)).fit()
print("nat yearly: Pearson chi2/df (homog. ratio)", round(g0.pearson_chi2/g0.df_resid,1))
# --- 2) Province x month: interannual stability (needs M>=50)
pm = mm.copy(); pm["r"] = pm.V/pm.M.replace(0,np.nan)
big = pm[pm.M>=100]
cv = big.groupby(["provincia_id","mes"]).r.agg(["count","mean","std"]); cv=cv[cv["count"]>=5]
cv["cv"]=cv["std"]/cv["mean"]
# expected Poisson CV for ratio ~ sqrt(1/V+...) approx sqrt(f/M)/f = 1/sqrt(fM)
exp = big.groupby(["provincia_id","mes"]).apply(lambda d: np.mean(1/np.sqrt(d.V.clip(lower=1))), include_groups=False)
cv["cv_poisson"]=exp.reindex(cv.index)
print("prov-mes con >=5 años M>=100:", len(cv), " CV interanual mediano", round(cv.cv.median(),3), " CV Poisson esperado mediano", round(cv.cv_poisson.median(),3), " razón", round((cv.cv/cv.cv_poisson).median(),1))
# Province x year ratio (all months) heterogeneity: GLM with prov*year
py = mm.groupby(["provincia_id","anio"])[["V","M"]].sum().reset_index(); py=py[py.M>0]
gA = smf.glm("V ~ provincia_id", data=py, family=sm.families.Poisson(), offset=np.log(py.M)).fit()
print("prov x año, modelo factor fijo por provincia: chi2/df =", round(gA.pearson_chi2/gA.df_resid,1), "(1 = Poisson puro)")
# Between-province spread of pooled ratios
pp = mm.groupby("provincia_id")[["V","M"]].sum(); pp["r"]=pp.V/pp.M
print(pp.sort_values("r").round(2).to_string())
# Month seasonality nat
mo = mm.groupby("mes")[["V","M"]].sum(); print("nat por mes", (mo.V/mo.M).round(2).to_dict())
# Trend per province (annual ratio Theil-Sen, % per year)
rows=[]
for p,d in py.groupby("provincia_id"):
    d=d[d.M>=200]
    if len(d)<8: continue
    rr=d.V/d.M; s=stats.theilslopes(rr.values,d.anio.values); rho=stats.spearmanr(d.anio,rr)
    rows.append((p,len(d),round(rr.mean(),2),round(100*s[0]/rr.mean(),2),round(rho.pvalue,3)))
t=pd.DataFrame(rows,columns=["prov","n","r_media","pend_%anual","p_spearman"]); print(t.to_string())
print("provincias con p<0.05:", (t.p_spearman<0.05).sum(), "de", len(t), "; pendiente mediana %/año", t["pend_%anual"].median())
