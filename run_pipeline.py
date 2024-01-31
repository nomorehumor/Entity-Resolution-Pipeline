import time
from er_vectorized_pipeline import assign_to_clusters, cluster_check, create_cluster_representatives_dict, create_deduplicated_dataset
from paths import OUTPUT_DIR
from pipeline.data_loading import load_two_publication_sets
from pipeline.deduplication import deduplicate_datasets

from pipeline.blocking import blocking
from pipeline.clustering import connected_components
from pipeline.matching import matching
from utils import convert_matches_to_indices_df


def run_entity_resolution(df_acm, df_dblp, blocking_function, matching_function, sim_threshold, blocking_params=None, matching_params={}):
    
    pipeline_start= time.time()
    pairs = blocking(df_acm, df_dblp, blocking_scheme=blocking_function, params=blocking_params)
    df_pairs = matching(df_acm, df_dblp, pairs, matching_function, weights=matching_params.get('matching_weights'))
    pipeline_end = time.time()
    print(f'Time needed for blocking and matching: {pipeline_end-pipeline_start}')

    # clusters = connected_components(df_pairs[df_pairs.similarity > sim_threshold])
    # deduplicate_datasets(df_acm, df_dblp, clusters)
    df_pairs = df_pairs[df_pairs["similarity"] > sim_threshold]
    indices_pairs_df = convert_matches_to_indices_df(df_acm, df_dblp, df_pairs)
    
    acm_clus, dblp_clus = assign_to_clusters(indices_df=indices_pairs_df)

    # use this function to write a file containing the papers belonging to each cluster
    cluster_check(acm_clus, dblp_clus, df_acm, df_dblp)

    representatives_dict = create_cluster_representatives_dict(dblp_clus, df_dblp)

    create_deduplicated_dataset(df_acm, acm_clus, representatives_dict, f"{OUTPUT_DIR}/deduplicated_acm_new.csv")
    create_deduplicated_dataset(df_dblp, dblp_clus, representatives_dict, f"{OUTPUT_DIR}/deduplicated_dblpp_new.csv")
    

if __name__ == "__main__":
    df_acm, df_dblp = load_two_publication_sets()

    run_entity_resolution(df_acm, df_dblp, "ngram_word_blocks", "cosine", 0.8, {"n":4})