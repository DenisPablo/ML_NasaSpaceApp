from common import *
pdd, op = load()
w = op.pivot(index="fecha", columns="satelite", values="n_nacional")
o = op.pivot(index="fecha", columns="satelite", values="operativo")
w.index=pd.to_datetime(w.index); o.index=pd.to_datetime(o.index)
# Suspicious "operational" days: n_nacional < 20% of centered 15-day rolling median AND other sensor shows normal activity
sus={}
for s in ["SNPP","Terra","Aqua","N20"]:
    x=w[s]; med=x.rolling(15,center=True,min_periods=7).median()
    # relative to concurrent other-sensor ratio: expected = other * rolling ratio
    other = "Aqua" if s in("SNPP","Terra","N20") else "Terra"
    if s=="SNPP": other="Aqua"
    rr=(x/w[other]).rolling(31,center=True,min_periods=10).median()
    expct=w[other]*rr
    flag=(o[s]==True)&(x< 0.2*expct)&(expct>=50)
    sus[s]=flag
    print(f"{s}: dias operativos {int(o[s].sum())}, caidos {int((o[s]==False).sum())}, 'operativos' con n<20% de lo esperado y esperado>=50: {int(flag.sum())}; con n_nacional<=5: {int(((o[s]==True)&(x<=5)).sum())}")
    if flag.sum(): print("   ejemplos:", [d.strftime('%Y-%m-%d') for d in flag[flag].index[-12:]])
# 2026 Aug-Nov daily view
v=w.loc["2026-07-25":"2026-11-13",["Terra","Aqua","SNPP","N20","N21"]]
print("\n== 2026-08..11: días con algún sat en 0 o muy bajo")
lowd=v[(v.fillna(0)<=0.15*v.rolling(15,center=True,min_periods=5).median()).any(axis=1)]
print(lowd.to_string())
# Effect on calibration: exclude suspicious days from valid days
sus_any = (sus["SNPP"]|sus["Terra"]|sus["Aqua"])
bad=set(sus_any[sus_any].index.date)
op2=op.copy(); op2.loc[op2.fecha.isin(bad)&op2.satelite.isin(["SNPP","Terra","Aqua"]),"operativo"]=False
mm=pd.read_parquet("mm.parquet")
mm2=CAL.monthly_counts(pdd,op2,"modis",(2012,2025))
f1,f2=CAL.fit_hierarchical(mm),CAL.fit_hierarchical(mm2)
print("\nDías sospechosos 2012-2025 excluidos:", sum(1 for d in bad if 2012<=d.year<=2025))
print("factor nacional base", round(f1.nat,4), " sin días sospechosos", round(f2.nat,4), " cambio %", round(100*(f2.nat/f1.nat-1),2))
x=f1.prov_m.merge(f2.prov_m,on=["provincia_id","mes"]); ch=100*(x.factor_y/x.factor_x-1)
print("cambio factores prov-mes: |mediana| %", round(ch.abs().median(),2), " max |%|", round(ch.abs().max(),2))
# Days excluded from calibration by the 'all operational' rule, and seasonality
days=CAL.valid_days(op,("SNPP","Terra","Aqua")); alld=pd.date_range("2012-02-01","2025-12-31").date
exc=[d for d in alld if d not in days]
print("días 2012-02..2025 excluidos de calibración:", len(exc), "de", len(alld), f"({100*len(exc)/len(alld):.1f}%)", " por mes:", pd.Series([d.month for d in exc]).value_counts().sort_index().to_dict())
# Focos lost: V in excluded days
pdd["fecha_ts"]=pd.to_datetime(pdd.fecha)
e=pdd[pdd.fecha.isin(set(exc))]
print("focos S-NPP en días excluidos:", int(e.n_snpp.sum()), " vs total 2012-2025:", int(pdd[(pdd.fecha_ts>='2012-02-01')&(pdd.fecha_ts<='2025-12-31')].n_snpp.sum()))
# --- API: ref down AND MODIS down -> estimate silently 0 (null != 0)
def down(s): return set(o.index[o[s]==False].date)
T,A,S=down("Terra"),down("Aqua"),down("SNPP")
pre=[d for d in pd.date_range("2000-11-01","2012-01-19").date]
t_only=[d for d in pre if d< pd.Timestamp("2002-07-04").date() and d in T]
ta=[d for d in pre if d>=pd.Timestamp("2002-07-04").date() and (d in T or d in A)]
both=[d for d in pre if d>=pd.Timestamp("2002-07-04").date() and (d in T and d in A)]
print(f"\nPre-VIIRS: días con Terra caído (2000-11..2002-07): {len(t_only)}; días 2002-07..2012-01 con Terra o Aqua caído: {len(ta)} (ambos caídos: {len(both)})")
ref=[d for d in pd.date_range("2012-01-20","2026-11-02").date if d in S]
rb=[d for d in ref if d in T or d in A]; rbb=[d for d in ref if d in T and d in A]
print(f"S-NPP caído 2012-01-20..2026-11-02: {len(ref)} días; de ellos con Terra o Aqua caído: {len(rb)}; ambos: {len(rbb)}")
print("  fechas S-NPP caído:", [d.isoformat() for d in ref])
# Expected lost VIIRS-equivalent focos on single-MODIS-down days pre-2012: use the other MODIS sat x (T+A)/sat ratio of the same month
pm=pdd.copy(); pm["ym"]=pd.to_datetime(pm.fecha).dt.to_period("M")
nat=pm.groupby("fecha")[["n_terra","n_aqua"]].sum(); nat.index=pd.to_datetime(nat.index)
loss=0; lossT=0
fct=3.33
for d in ta:
    dt=pd.Timestamp(d); 
    if dt not in nat.index: 
        continue
    mon=nat.loc[dt-pd.Timedelta(days=15):dt+pd.Timedelta(days=15)]
    if d in T and d not in A: loss += nat.loc[dt,"n_aqua"]*(mon.n_terra.sum()/max(mon.n_aqua.sum(),1))
    elif d in A and d not in T: loss += nat.loc[dt,"n_terra"]*(mon.n_aqua.sum()/max(mon.n_terra.sum(),1))
print(f"MODIS faltante estimado (2002-07..2012-01, días con un solo MODIS caído): {loss:.0f} focos MODIS ≈ {loss*fct:.0f} VIIRS-equivalentes; total MODIS del período: {int(nat.loc['2002-07-04':'2012-01-19'].sum().sum())}")
# months most affected
yy=pd.Series([pd.Timestamp(d).to_period('M') for d in ta]).value_counts().head(8); print("meses con más días con un MODIS caído:", {str(k):int(v) for k,v in yy.items()})
# Terra-down days by year
print("Terra caído por año:", pd.Series([d.year for d in T]).value_counts().sort_index().to_dict())
print("Aqua caído por año:", pd.Series([d.year for d in A]).value_counts().sort_index().to_dict())
