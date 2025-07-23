from pyspark.sql import SparkSession
import pandas as pd

print("="*60)
print("TELECOM INFRASTRUCTURE ANALYSIS USING Pyspark RDD MAPREDUCE")
print("="*60)
print("\nSTEP 1: Initializing Spark Session...\n")
spark = SparkSession.builder.appName("OpenCelliD MapReduce Revised").getOrCreate()
print("Spark session created successfully!")

# ----------------------------------------------------------
print("\nSTEP 2: Loading and Preprocessing Dataset...\n")
df = pd.read_csv("310.csv")
df.columns = [
    "radio", "mcc", "mnc", "area", "cell", "unit",
    "lon", "lat", "range", "samples", "changeable",
    "created", "updated", "unknown"
]
df.drop(columns=["unit", "unknown"], inplace=True)
spark_df = spark.createDataFrame(df)
print("DataFrame converted to Spark DataFrame.")

# ----------------------------------------------------------
print("\nSTEP 3: Counting Total Number of Towers (Rows)")
row_count = spark_df.rdd.count()
print(f"RESULT: Total number of towers (rows): {row_count}")

# ----------------------------------------------------------
print("\nSTEP 4: Distinct Technologies (Radio Types)")
tech_rdd = spark_df.select("radio").rdd.map(lambda row: row["radio"])
unique_tech = tech_rdd.distinct().collect()
print(f"RESULT: Distinct technologies: {unique_tech} (Total: {len(unique_tech)})")

# ----------------------------------------------------------
print("\nSTEP 5: Count Towers for Each Technology (MapReduce)")
tech_count_rdd = spark_df.rdd.map(lambda row: (row["radio"], 1))
tech_counts = tech_count_rdd.reduceByKey(lambda a, b: a + b).collect()
print("Tower count by technology:")
for tech, count in tech_counts:
    print(f"  - {tech}: {count}")

# ----------------------------------------------------------
print("\nSTEP 6: Find the Region with the Most Towers (rounded lat/lon)")
region_rdd = spark_df.rdd.map(lambda row: ((round(row["lat"]), round(row["lon"])), 1))
region_counts = region_rdd.reduceByKey(lambda a, b: a + b)
region_counts_list = region_counts.collect()
region_counts_sorted = sorted(region_counts_list, key=lambda x: x[1], reverse=True)
max_region = region_counts_sorted[0]
print(f"RESULT: Region with most towers: (Lat: {max_region[0][0]}, Lon: {max_region[0][1]})")
print(f"Number of towers in this region: {max_region[1]}")
print("Top 5 regions by tower count:")
for idx, (reg, cnt) in enumerate(region_counts_sorted[:5]):
    print(f"  {idx+1}. (Lat: {reg[0]}, Lon: {reg[1]}) - {cnt} towers")

# ----------------------------------------------------------
print("\nSTEP 7: Average, Min, Max Estimated Coverage Radius (`range`)")
range_rdd = spark_df.select("range").rdd.map(lambda row: row["range"])
range_sum = range_rdd.reduce(lambda a, b: a + b)
range_count = range_rdd.count()
average_range = range_sum / range_count if range_count else 0
min_range = range_rdd.min()
max_range = range_rdd.max()
print(f"RESULT: Average estimated coverage radius (meters): {average_range:.2f}")
print(f"RESULT: Minimum estimated coverage radius (meters): {min_range}")
print(f"RESULT: Maximum estimated coverage radius (meters): {max_range}")

# ----------------------------------------------------------
print("\nSTEP 8: Unique Cell IDs")
cellid_rdd = spark_df.select("cell").rdd.map(lambda row: row["cell"])
unique_cells = cellid_rdd.distinct().count()
print(f"RESULT: Number of unique cell IDs: {unique_cells}")

# ----------------------------------------------------------
print("\nSTEP 9: Top 5 Areas by Average Coverage Radius")
region_range_rdd = spark_df.rdd.map(lambda row: ((round(row["lat"]), round(row["lon"])), (row["range"], 1)))
region_range_avg = region_range_rdd.reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
                                   .mapValues(lambda x: round(x[0]/x[1], 2)) \
                                   .collect()
region_range_avg_sorted = sorted(region_range_avg, key=lambda x: x[1], reverse=True)
print("Top 5 regions by average coverage radius:")
for idx, (region, avg_range) in enumerate(region_range_avg_sorted[:5]):
    print(f"  {idx+1}. (Lat: {region[0]}, Lon: {region[1]}) - Avg Radius: {avg_range} meters")

# ----------------------------------------------------------
print("\nSTEP 10: Towers with the Most Samples")
tower_sample_rdd = spark_df.rdd.map(lambda row: (f"{row['area']}_{row['cell']}", row["samples"]))
tower_sample_sum = tower_sample_rdd.reduceByKey(lambda a, b: a + b) \
                                   .collect()
tower_sample_sorted = sorted(tower_sample_sum, key=lambda x: x[1], reverse=True)
print("Top 5 towers by sample count:")
for idx, (tower_id, samples) in enumerate(tower_sample_sorted[:5]):
    print(f"  {idx+1}. Tower ID {tower_id}: {samples} samples")

print("\n" + "="*60)
print("END OF MAPREDUCE DEMONSTRATION SCRIPT")
print("="*60)
