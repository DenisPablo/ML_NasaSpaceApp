from common import *
from scipy import stats
from math import ceil
pdd, op = load()
pub=pub_factors(); F={(r.provincia_id,r.mes,r.tipo):r.factor for r in pub.itertuples()}
o=op.pivot(index="fecha",columns="satelite",values="operativo")
def ref(d):
    if pd.Timestamp("2012-01-20").date()<=d<=pd.Timestamp("2026-11-02").date(): return "SNPP"
    if d>=pd.Timestamp("2026-11-03").date(): return "N20"
    return None
x=pdd.copy(); x["ref"]=x.fecha.map(ref)
x["refop"]=[bool(o.at[d,r]) if isinstance(r,str) and d in o.index and pd.notna(o.at[d,r]) else False for d,r in zip(x.fecha,x.ref)]
x["obs"]=np.where(x.refop, np.where(x.ref.eq("SNPP"),x.n_snpp,x.n_n20),0)
pre=x.fecha<pd.Timestamp("2002-07-04").date()
x["et"]=np.where(~x.refop & pre, x.n_terra,0); x["em"]=np.where(~x.refop & ~pre, x.n_terra+x.n_aqua,0)
x["mes"]=pd.to_datetime(x.fecha).dt.month; x["ym"]=pd.to_datetime(x.fecha).dt.to_period("M")
x["est"]=x.et*[F[(p,m,"terra")] for p,m in zip(x.provincia_id,x.mes)]+x.em*[F[(p,m,"modis")] for p,m in zip(x.provincia_id,x.mes)]
x["arm"]=x.obs+x.est
nat=x.groupby("ym")[["arm","obs","est","et","em"]].sum()
nat=nat[nat.index<=pd.Period("2026-11")]
ann=nat.groupby(nat.index.year).arm.sum(); ann=ann[(ann.index>=2001)&(ann.index<=2025)]
print("Anual armonizado nacional:", ann.round(0).astype(int).to_dict())
ts=stats.theilslopes(ann.values,ann.index.values); print("Theil-Sen %/año:", round(100*ts[0]/ann.median(),2), " Kendall tau", stats.kendalltau(ann.index,ann.values))
e1=ann.loc[2003:2011]; e2=ann.loc[2012:2025]
print("mediana anual 2003-2011 (estimado):", int(e1.median()), " 2012-2025 (observado):", int(e2.median()), " MWU p=", round(stats.mannwhitneyu(e1,e2).pvalue,3))
# monthly era comparison
nat["mes"]=nat.index.month; nat["y"]=nat.index.year
cmp=[]
for m in range(1,13):
    a=nat[(nat.mes==m)&(nat.y.between(2003,2011))].arm; b=nat[(nat.mes==m)&(nat.y.between(2012,2025))].arm
    cmp.append((m,int(a.median()),int(b.median()),round(b.median()/a.median(),2)))
print("mediana por mes (2003-11 est vs 2012-25 obs, razón):", cmp)
# Anomaly category replica
def cu(s,p): return s[ceil(len(s)*p)-1]
def cat(v,s):
    if v<cu(s,.1): return "mb"
    if v<cu(s,.25): return "b"
    if v>cu(s,.9): return "ma"
    if v>cu(s,.75): return "a"
    return "n"
full=nat[nat.index<=pd.Period("2026-10")]  # complete months
agree=tot=0; ch_era=[]
for p,r in full.iterrows():
    if p.year<2012: continue
    s_all=sorted(round(v,1) for q,v in full.arm.items() if q!=p and q.month==p.month)
    s_obs=sorted(round(v,1) for q,v in full.arm.items() if q!=p and q.month==p.month and q.year>=2012)
    c1,c2=cat(round(r.arm,1),s_all),cat(round(r.arm,1),s_obs); tot+=1; agree+=c1==c2
    if c1!=c2: ch_era.append(f"{p}:{c1}->{c2}")
print(f"Categorías 2012-2026: todos los años vs solo años observados: coinciden {agree}/{tot} ({100*agree/tot:.0f}%) ; n años solo-obs = 13-14")
print("  ejemplos de cambio:", ch_era[:12])
# partial month contamination (API includes partial 2026-11 in reference sample)
nov_partial=nat.loc[pd.Period("2026-11")].arm
chg=[]
for p,r in full.iterrows():
    if p.month!=11: continue
    s_clean=sorted(round(v,1) for q,v in full.arm.items() if q!=p and q.month==11)
    s_api=sorted(s_clean+[round(nov_partial,1)])
    a,b=cat(round(r.arm,1),s_clean),cat(round(r.arm,1),s_api)
    if a!=b: chg.append(f"{p}:{a}->{b}")
print(f"Nov-2026 parcial (1-13) armonizado={nov_partial:.0f} vs mediana Nov={np.median(full[full.mes==11].arm):.0f}; noviembres con categoría alterada por incluirlo: {chg}")
print("años por mes calendario (meses completos):", full.groupby("mes").size().to_dict())
# months with large frac estimated in observed era
nat.to_parquet("nat.parquet")
# July 2025 SNPP partial gap impact
j=nat.loc[pd.Period("2025-07")]; print("2025-07 armonizado nacional:", round(j.arm), " (gap parcial S-NPP 4-7 jul ≈ 1.000 focos faltantes por NOAA-20)")
