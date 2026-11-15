from common import *
pdd, op = load()
w = op.pivot(index="fecha", columns="satelite", values="n_nacional"); o = op.pivot(index="fecha", columns="satelite", values="operativo")
w.index=pd.to_datetime(w.index); o.index=pd.to_datetime(o.index)
x=w.loc["2018-04-01":"2026-11-02"].copy(); ox=o.loc[x.index]
x=x[ox.SNPP & ox.N20]
print("razón global SNPP/N20 2018-04..2026-11-02 (días ambos operativos):", round(x.SNPP.sum()/x.N20.sum(),3))
r=x.SNPP/x.N20
g=x[(x.N20>=50)&(r<0.5)]
print("días S-NPP 'operativo' con SNPP < 0.5×N20 (N20>=50):", len(g), " focos faltantes (N20-SNPP):", int((g.N20-g.SNPP).sum()))
print("por año:", g.groupby(g.index.year).apply(lambda d: (len(d), int((d.N20-d.SNPP).sum()))).to_dict())
print(g.assign(r=r.round(2)).tail(15).to_string())
# simétrico: N20 parcial
g2=x[(x.SNPP>=50)&(x.N20/x.SNPP<0.5)]; print("días N20 < 0.5×SNPP:", len(g2), " faltantes", int((g2.SNPP-g2.N20).sum()))
# NOAA-20 period (2026-11-03..) vs N21
y=w.loc["2026-11-03":]; print("N20 vs N21 desde 2026-11-03:", y[["N20","N21"]].sum().to_dict())
# Terra 'caído': do the other sensors see fire those days?
for s,oth in (("Terra","Aqua"),("Aqua","Terra")):
    d=w[(o[s]==False)&(w.index>=pd.Timestamp("2003-01-01"))]
    print(f"{s} caído (n=0) desde 2003: {len(d)} días; mediana de focos {oth} ese día: {d[oth].median()}, con {oth}>=20: {(d[oth]>=20).sum()}, con {oth}<=3: {(d[oth]<=3).sum()}")
