from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as pyspark_round, avg, stddev, count, min, max, sum, concat_ws

# 1. Start Spark session
spark = SparkSession.builder.appName("OpenCelliD Advanced Spark Analysis").getOrCreate()

# 2. Load your dataset (update path if needed)
import pandas as pd
df = pd.read_csv("310.csv")

# 3. Assign correct column names (based on your dataset)
df.columns = [
    "radio", "mcc", "mnc", "area", "cell", "unit",
    "lon", "lat", "range", "samples", "changeable",
    "created", "updated", "unknown"
]
df.drop(columns=["unit", "unknown"], inplace=True)

# 4. Create Spark DataFrame
spark_df = spark.createDataFrame(df)

# 1. Technology Distribution: Towers, Average Radius, Sample Sum per Technology
print("\n--- Technology Distribution (Towers, Avg Radius, Total Samples) ---")
tech_stats = spark_df.groupBy("radio").agg(
    count("*").alias("tower_count"),
    avg("range").alias("avg_radius"),
    stddev("range").alias("stddev_radius"),
    sum("samples").alias("total_samples")
)
tech_stats.show()

# 2. Region Summary: (rounded lat/lon) - Tower Count, Avg Radius, Avg Samples
print("\n--- Region Summary: Tower Count, Avg Radius, Avg Samples ---")
df_region = spark_df.withColumn("lat_r", pyspark_round(col("lat")).cast("int")) \
                    .withColumn("lon_r", pyspark_round(col("lon")).cast("int"))
region_stats = df_region.groupBy("lat_r", "lon_r").agg(
    count("*").alias("tower_count"),
    avg("range").alias("avg_radius"),
    avg("samples").alias("avg_samples")
)
region_stats.orderBy(col("tower_count").desc()).show(10)

# 3. Technology per Region: LTE/UMTS/NR/GSM towers by region (top 10 regions)
print("\n--- Technology per Region (LTE/UMTS/NR/GSM counts) ---")
tech_per_region = df_region.groupBy("lat_r", "lon_r", "radio").count()
tech_per_region.orderBy(col("count").desc()).show(20)

# 4. Largest and Smallest Coverage Regions
print("\n--- Largest and Smallest Average Coverage Radius by Region ---")
region_coverage = region_stats.orderBy(col("avg_radius").desc())
region_coverage.show(5)
region_coverage.orderBy(col("avg_radius").asc()).show(5)


print("\n--- Top 10 Towers by Total Samples Collected ---")
spark_df.withColumn(
    "tower_id", 
    concat_ws("_", col("area").cast("string"), col("cell").cast("string"))
).groupBy("tower_id") \
 .agg(sum("samples").alias("total_samples")) \
 .orderBy(col("total_samples").desc()) \
 .show(10)


## 6. Technology Share in Top-Density Regions
print("\n--- Technology Share in Top Density Regions ---")
top_regions = region_stats.orderBy(col("tower_count").desc()).limit(5).select("lat_r", "lon_r")
df_top_regions = df_region.join(top_regions, on=["lat_r", "lon_r"], how="inner")
df_top_regions.groupBy("lat_r", "lon_r", "radio").count().orderBy("lat_r", "lon_r", col("count").desc()).show(30)

# 7. Under-served Region Candidates: (few towers, small avg radius)
print("\n--- Under-Served Region Candidates (tower_count < 10, avg_radius < 2000) ---")
region_stats.filter((col("tower_count") < 10) & (col("avg_radius") < 2000)) \
            .orderBy(col("avg_radius").asc()) \
            .show(10)

# 8. Coverage Range Stats: Min, Max, Avg, Stddev for the whole dataset
print("\n--- Overall Coverage Range Stats ---")
spark_df.select(
    min("range").alias("min_radius"),
    max("range").alias("max_radius"),
    avg("range").alias("avg_radius"),
    stddev("range").alias("stddev_radius")
).show()

from pyspark.sql.functions import year, from_unixtime, lit, when

# ----- 1. Temporal Analysis: Tower Count per Year, per Technology -----
print("\n--- Temporal Analysis: Towers Added Per Year By Technology ---")
# Assume 'created' is a Unix timestamp (int); if not, adjust accordingly.
df_with_year = spark_df.withColumn("year", year(from_unixtime(col("created"))))
towers_per_year_tech = df_with_year.groupBy("year", "radio").count().orderBy("year", "radio")
towers_per_year_tech.show(20)

# Optional: show just recent years for a cleaner view
print("\n--- Recent 10 Years, Towers Added Per Year By Technology ---")
recent_years = df_with_year.filter(col("year") >= 2015)
recent_years.groupBy("year", "radio").count().orderBy("year", "radio").show(40)

# ----- 2. Operator Mapping (MCC+MNC => Provider) -----
print("\n--- Operator Share By Technology ---")
# Example mapping for demo purposes
operator_dict = {
    (310, 260): "T-Mobile US",
    (310, 410): "AT&T US",
    (310, 120): "Sprint",
    (311, 480): "Verizon",
    # Add more mappings as needed
}
# Create a UDF to map (mcc, mnc) tuple to operator name
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def mcc_mnc_to_operator(mcc, mnc):
    return operator_dict.get((int(mcc), int(mnc)), "Unknown")

mcc_mnc_udf = udf(mcc_mnc_to_operator, StringType())

df_with_operator = spark_df.withColumn("operator", mcc_mnc_udf(col("mcc"), col("mnc")))

# Operator market share by technology
op_share = df_with_operator.groupBy("operator", "radio").count().orderBy(col("count").desc())
op_share.show(20)

# Operator share by region (optional, for detailed mapping)
# op_region_share = df_with_operator.withColumn("lat_r", pyspark_round(col("lat")).cast("int")) \
#                                   .withColumn("lon_r", pyspark_round(col("lon")).cast("int")) \
#                                   .groupBy("lat_r", "lon_r", "operator").count() \
#                                   .orderBy(col("count").desc())
# op_region_share.show(20)
