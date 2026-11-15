from common import con, D
q=lambda s: print(con.sql(s))
for t in ["factores_provincia_mes","factores_region_mes","rangos","satelite_operativo","dim_provincias"]:
    q(f"DESCRIBE SELECT * FROM '{D}/{t}.parquet'")
q(f"SELECT * FROM '{D}/rangos.parquet'")
q(f"SELECT tipo, nivel_origen, count(*), min(factor), median(factor), max(factor) FROM '{D}/factores_provincia_mes.parquet' GROUP BY ALL ORDER BY ALL")
q(f"SELECT tipo, region, round(min(factor),2) mn, round(max(factor),2) mx, round(avg(factor),2) av FROM '{D}/factores_region_mes.parquet' GROUP BY ALL ORDER BY ALL")
q(f"SELECT satelite, min(fecha), max(fecha), count(*) dias, sum(CASE WHEN NOT operativo THEN 1 ELSE 0 END) caidos FROM '{D}/satelite_operativo.parquet' GROUP BY 1 ORDER BY 1")
q(f"SELECT provincia_id, nombre, region FROM '{D}/dim_provincias.parquet' ORDER BY 1")
q(f"SELECT count(*), min(fecha), max(fecha) FROM read_parquet('{D}/conteos_celda_dia/*/*.parquet', hive_partitioning=1)")
