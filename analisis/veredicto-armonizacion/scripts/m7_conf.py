from common import *
from scipy import stats
from math import ceil
nat=pd.read_parquet("nat.parquet")
# category distribution 2012-2025 with all-years reference (API behavior)
def cu(s,p): return s[ceil(len(s)*p)-1]
full=nat[nat.index<=pd.Period("2026-10")]
cats=[]
for p,r in full.iterrows():
    s=sorted(v for q,v in full.arm.items() if q!=p and q.month==p.month)
    v=r.arm; c="mb" if v<cu(s,.1) else "b" if v<cu(s,.25) else "ma" if v>cu(s,.9) else "a" if v>cu(s,.75) else "n"
    cats.append((p.year,c))
c=pd.DataFrame(cats,columns=["y","c"]); c["era"]=np.where(c.y>=2012,"2012+","2001-2011")
print("distribución de categorías por era (%):"); print(pd.crosstab(c.era,c.c,normalize="index").mul(100).round(0).to_string())
# raw MODIS annual (all days) Terra, Aqua
D_=D
raw=con.sql(f"""SELECT year(fecha) y, sum(n_terra) t, sum(n_aqua) a, sum(n_snpp) s FROM read_parquet('{D_}/conteos_celda_dia/*/*.parquet', hive_partitioning=1) GROUP BY 1 ORDER BY 1""").df().set_index("y")
raw["a_share"]=(raw.a/(raw.t+raw.a)).round(3); raw["s_over_m"]=(raw.s/(raw.t+raw.a)).round(2)
print(raw.to_string())
m=raw.t+raw.a
print("MODIS crudo mediana 2003-2011:", int(m.loc[2003:2011].median()), " 2012-2025:", int(m.loc[2012:2025].median()), " razón", round(m.loc[2012:2025].median()/m.loc[2003:2011].median(),3))
print("Aqua/Terra por año (ratio):", (raw.a/raw.t).loc[2003:2025].round(2).to_dict())
r=(raw.a/raw.t).loc[2003:2025]; print("Theil-Sen Aqua/Terra por año:", round(stats.theilslopes(r.values,r.index.values)[0],4), " Kendall p", round(stats.kendalltau(r.index,r.values).pvalue,4))
# V/Terra and V/Aqua per year on valid days
pdd,op=load(); days=CAL.valid_days(op,("SNPP","Terra","Aqua"))
x=pdd[pdd.fecha.isin(days)].copy(); x["y"]=pd.to_datetime(x.fecha).dt.year; x=x[x.y.between(2012,2025)]
g=x.groupby("y")[["n_snpp","n_terra","n_aqua"]].sum()
g["V/T"]=(g.n_snpp/g.n_terra).round(2); g["V/A"]=(g.n_snpp/g.n_aqua).round(2); print(g[["V/T","V/A"]].T.to_string())
for col in ["V/T","V/A"]:
    print(col,"Kendall p", round(stats.kendalltau(g.index,g[col]).pvalue,4), " Theil %/año", round(100*stats.theilslopes(g[col].values,g.index.values)[0]/g[col].median(),2))
# Terra-only path vs MODIS path in pre-VIIRS era 2003-2011 (consistency of the Terra->Terra+Aqua switch)
pub=pub_factors(); F={(r.provincia_id,r.mes,r.tipo):r.factor for r in pub.itertuples()}
y=pdd.copy(); y["ts"]=pd.to_datetime(y.fecha); y=y[(y.ts>="2002-07-04")&(y.ts<"2012-01-20")]; y["mes"]=y.ts.dt.month
y["est_terra"]=y.n_terra*[F[(p,mm,"terra")] for p,mm in zip(y.provincia_id,y.mes)]
y["est_modis"]=(y.n_terra+y.n_aqua)*[F[(p,mm,"modis")] for p,mm in zip(y.provincia_id,y.mes)]
a=y.groupby(y.ts.dt.year)[["est_terra","est_modis"]].sum(); a["ratio_T/M"]=(a.est_terra/a.est_modis).round(3)
print("2002-07..2012-01: estimado vía solo Terra / vía Terra+Aqua por año:", a["ratio_T/M"].to_dict(), " total:", round(a.est_terra.sum()/a.est_modis.sum(),3))
# Same check in calibration-period years 2020-2025 (Terra drift; terra factors from 2012-2019)
z=x.copy(); z["mes"]=pd.to_datetime(z.fecha).dt.month
z["et"]=z.n_terra*[F[(p,mm,"terra")] for p,mm in zip(z.provincia_id,z.mes)]
b=z.groupby("y")[["et","n_snpp"]].sum(); print("factor Terra(2012-19) aplicado: sesgo % por año:", (100*(b.et/b.n_snpp-1)).round(1).to_dict())
