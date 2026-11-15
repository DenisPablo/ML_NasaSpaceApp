from common import *
rg=rangos()
mt=pd.read_parquet("mt.parquet")
out=[]
for y in sorted(mt.anio.unique()):
    tr,te=mt[mt.anio!=y],mt[mt.anio==y].copy(); te["pred"]=CAL.predict(te,CAL.fit_hierarchical(tr)); out.append(te)
d=pd.concat(out); ok,_,_=cover(d.M.values,d.V.values,d.pred.values,rg)
print("Terra sola LOYO 2012-2019, rangos publicados: cobertura", round(100*ok.mean(),1), " error prov-mes", round(100*np.abs(d.pred-d.V).sum()/d.V.sum(),1))
print(pd.Series(ok).groupby(pd.cut(d.M.values,[-1,0,5,20,100,500,1e12]).astype(str)).mean().mul(100).round(1).to_dict())
l=pd.read_parquet("loyo200.parquet"); n=l.groupby(["anio","mes"])[["V","pred","M"]].sum(); e=(n.pred-n.V).abs()/n.V
print("Nacional mensual 100% estimado (LOYO): err mediano %", round(100*e.median(),1), " p90", round(100*e.quantile(.9),1))
# 'media' class at province level: M>=21 LOYO error
x=l[l.M>=21]; e2=(x.pred-x.V).abs()/x.V.replace(0,np.nan); print("prov-mes M>=21: err mediano %", round(100*e2.median(),1), " p90", round(100*e2.quantile(.9),1), "; M 6-20:", round(100*((l[(l.M>=6)&(l.M<=20)].pred-l[(l.M>=6)&(l.M<=20)].V).abs()/l[(l.M>=6)&(l.M<=20)].V.replace(0,np.nan)).median(),1))
