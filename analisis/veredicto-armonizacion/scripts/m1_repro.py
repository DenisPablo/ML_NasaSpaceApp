from common import *
pdd, op = load()
mm = CAL.monthly_counts(pdd, op, "modis", (C.CALIB_START.year, C.CALIB_END.year))
mt = CAL.monthly_counts(pdd, op, "terra", C.TERRA_CALIB_YEARS)
fm, ft = CAL.fit_hierarchical(mm), CAL.fit_hierarchical(mt)
pub = pub_factors()
for tipo, fit in (("modis", fm), ("terra", ft)):
    x = fit.prov_m.merge(pub[pub.tipo==tipo], on=["provincia_id","mes"])
    print(tipo, "nat", round(fit.nat,4), "max|diff|", float((x.factor_x-x.factor_y).abs().max()))
rg = CAL.calibrate_ranges(mm); print(rg)
print("cov in-sample", CAL.range_coverage(mm, fm, rg.drop(columns="n_meses")))
mm.to_parquet("mm.parquet"); mt.to_parquet("mt.parquet")
