import time
from scripts.er_vectorized_pipeline import assign_to_clusters, cluster_check, create_cluster_representatives_dict, create_deduplicated_dataset
from paths import OUTPUT_DIR
from pipeline.data_loading import load_two_publication_sets

from pipeline.blocking import blocking
from pipeline.clustering import connected_components, deduplicate_datasets
from pipeline.matching import matching


def run_entity_resolution(df_acm, df_dblp, blocking_function, matching_function, sim_threshold, blocking_params=None, matching_params={}):
    print(f'Running entity resolution with blocking function {blocking_function} and matching function {matching_function}')
    pipeline_start= time.time()
    # Blocking
    pairs = blocking(df_acm, df_dblp, blocking_scheme=blocking_function, params=blocking_params)
    
    # Matching
    df_pairs = matching(df_acm, df_dblp, pairs, matching_function, weights=matching_params.get('matching_weights'))
    df_pairs[["index_acm", "paperId_acm", "index_dblp", "paperId_dblp"]].to_csv(f'{OUTPUT_DIR}/Matched_Entities.csv', index=False)
    pipeline_end = time.time()
    print(f'Time needed for blocking and matching: {pipeline_end-pipeline_start}')

    # Create clusters
    clusters = connected_components(df_pairs[df_pairs.similarity > sim_threshold])
    df_acm_dedup, df_dblp_dedup = deduplicate_datasets(df_acm, df_dblp, clusters)
    
    # Save deduplicated datasets
    df_acm_dedup.to_csv(f'{OUTPUT_DIR}/ACM_deduplicated.csv')
    df_dblp_dedup.to_csv(f'{OUTPUT_DIR}/DBLP_deduplicated.csv')    

if __name__ == "__main__":
    df_acm, df_dblp = load_two_publication_sets()

    run_entity_resolution(df_acm, df_dblp, "ngram_word_blocks", "cosine", 0.8, {"n":4})