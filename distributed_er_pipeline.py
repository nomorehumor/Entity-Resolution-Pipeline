import time

import Levenshtein

from pipeline.clustering import connected_components
from paths import ACM_DATASET_FILE, DBLP_DATASET_FILE, OUTPUT_DIR

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, NGram
from pyspark.sql.functions import regexp_replace, concat_ws, udf, col, explode, collect_list, size, lower, concat, \
    lit, trim
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, FloatType
from pyspark.sql.functions import monotonically_increasing_id

spark = SparkSession.builder \
    .appName('ERPipeline') \
    .master("local[10]") \
    .getOrCreate()


def read_file_spark(filename, dataset_origin):
    schema = StructType([
        StructField("paperId", StringType(), True),
        StructField("title", StringType(), True),
        StructField("authors", StringType(), True),
        StructField("venue", StringType(), True),
        StructField("year", IntegerType(), True)
    ])

    df = (spark.read.option("sep", "|").option("header", True)
          .option("inferSchema", True)
          .option("maxPartitionBytes", "134217728")
          .csv(filename, schema=schema))

    df = df.na.fill("")

    df = preprocess_column(df, 'title', 'cleaned_title')
    df = preprocess_column(df, 'authors', 'cleaned_authors')

    df = df.withColumn("Combined", concat(col("cleaned_list_title"), col("cleaned_list_authors")))
    df = df.withColumn("index", monotonically_increasing_id())
    drop_cols = ["cleaned_list_title", "cleaned_list_authors"]
    df = df.drop(*drop_cols)

    column_names = [f"{col}_{dataset_origin}" for col in df.columns]
    df = df.toDF(*column_names)

    df = df.repartition(10)
    return df


def preprocess_column(df, input_column, output_column):
    df = df.withColumn("cleaned",
                       trim(regexp_replace(regexp_replace(lower(col(input_column)), "[^a-z0-9]", " "), r" +", " ")))

    tokenizer = Tokenizer(inputCol='cleaned', outputCol=f"cleaned_list_{input_column}")
    df = tokenizer.transform(df)

    df = df.withColumn(output_column, concat_ws(" ", col(f"cleaned_list_{input_column}")))

    drop_cols = [f"words_{input_column}", 'cleaned']
    df = df.drop(*drop_cols)

    return df


def load_two_publication_sets_spark():
    # Load ACM and DBLP datasets
    df_acm = read_file_spark(ACM_DATASET_FILE, "acm")
    df_dblp = read_file_spark(DBLP_DATASET_FILE, "dblp")

    return df_acm, df_dblp


def blocking_spark(df_acm, df_dblp):
    blocks1 = create_ngram_word_blocks_spark(df_acm, "Combined_acm", "index_acm", 3)
    blocks2 = create_ngram_word_blocks_spark(df_dblp, "Combined_dblp", "index_dblp", 3)

    candidate_pairs_set = get_candidate_pairs_between_blocks_spark(blocks1, blocks2, 'ngram', 'ids')

    return candidate_pairs_set


def create_ngram_word_blocks_spark(df, column, id_col, n):
    ngram = NGram(n=n, inputCol=column, outputCol="ngrams")
    df = ngram.transform(df)

    exploded_df = df.select(col(id_col), explode(col("ngrams")).alias("ngram"))
    repartitioned_df = exploded_df.repartition(10)
    blocks = repartitioned_df.groupBy("ngram").agg(collect_list(id_col).alias("ids"))
    blocks = blocks.repartition(10)

    return blocks


def get_candidate_pairs_between_blocks_spark(blocks1, blocks2, column, id_col):
    ids1 = blocks1.select(col(column).alias("words_acm"), explode(col(id_col)).alias("id_acm"))
    ids2 = blocks2.select(col(column).alias("words_dblp"), explode(col(id_col)).alias("id_dblp"))

    candidate_pairs = ids1.join(ids2, ids1["words_acm"] == ids2["words_dblp"]).select([col('id_acm'), col('id_dblp')])

    return candidate_pairs.dropDuplicates()


def levenshtein_matching(df_acm, df_dblp, pairs, threshold, weights=[0.33, 0.33, 0.33]):
    joined_sdf = pairs.join(df_acm, col("id_acm") == col("index_acm"))
    joined_sdf = joined_sdf.join(df_dblp, col("id_dblp") == col("index_dblp"))

    levenshtein_udf = udf(
        lambda s1, s2: 1 - Levenshtein.distance(s1, s2) / max(len(s1), len(s2)) if max(len(s1), len(s2)) > 0 else 0,
        FloatType())

    title_sim = levenshtein_udf(col("cleaned_title_acm"), col("cleaned_title_dblp"))
    authors_sim = levenshtein_udf(col("cleaned_authors_acm"), col("cleaned_authors_dblp"))
    year_sim = (col("year_acm") == col("year_dblp")).cast("int")

    similarity_col = weights[0] * title_sim + weights[1] * authors_sim + weights[2] * year_sim

    result_sdf = joined_sdf.withColumn("similarity", similarity_col)
    cols = [col('index_acm'), col('index_dblp'), col('similarity'), col('cleaned_title_acm'), col('cleaned_title_dblp')]
    matched_entities = result_sdf.select(cols).filter(col("similarity") > threshold)

    return matched_entities


def create_undirected_bipartite_graph_distributed(matched_pairs):
    edges_df = matched_pairs.select(
        concat(lit("1_"), col("id_acm")).alias("node1"),
        concat(lit("2_"), col("id_dblp")).alias("node2")
    )

    undirected_edges_df = edges_df.union(edges_df.select("node2", "node1"))

    graph_rdd = undirected_edges_df.rdd.map(lambda row: (row[0], row[1]))
    graph = graph_rdd.groupByKey().mapValues(list).collectAsMap()

    return graph


def deduplicate_datasets_spark(df_acm, df_dblp, clusters):
    idx_acm, idx_dblp = [], []

    for key in clusters.keys():
        if len(clusters[key]) > 2:
            id_acm = [int(el[2:]) for el in clusters[key] if el.startswith('1')]
            idx_acm.extend(id_acm[1:])
            id_dblp = [int(el[2:]) for el in clusters[key] if el.startswith('2')]
            idx_dblp.extend(id_dblp[1:])

    df_acm = df_acm.filter(~col("id_acm").cast(IntegerType()).isin(idx_acm))
    df_dblp = df_dblp.filter(~col("id_dblp").cast(IntegerType()).isin(idx_dblp))

    return df_acm, df_dblp


def pipeline():
    sim_threshold = 0.8

    pipeline_start = time.time()
    df_acm, df_dblp = load_two_publication_sets_spark()
    candidate_pairs = blocking_spark(df_acm, df_dblp)
    matched_entities = levenshtein_matching(df_acm, df_dblp, candidate_pairs, sim_threshold)
    clusters = connected_components(matched_entities)
    dedupe_acm, dedupe_dblp = deduplicate_datasets_spark(df_acm, df_dblp, clusters)

    pipeline_end = time.time()
    print(f'Pipeline execution time: {pipeline_end - pipeline_start}')

    matched_entities.coalesce(1).write.option("header", "true") \
        .mode("overwrite") \
        .csv("output/distributed_matched_entities")

    dedupe_acm.coalesce(1).write.csv(f"{OUTPUT_DIR}/CM_deduplicated_distributed.csv", header=True,
                                     mode="overwrite")
    dedupe_dblp.coalesce(1).write.csv(f"{OUTPUT_DIR}/DBLP_deduplicated_distributed.csv", header=True,
                                      mode="overwrite")

    return clusters
