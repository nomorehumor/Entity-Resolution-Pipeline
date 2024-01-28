import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

from blocking import create_ngram_word_blocks
from data_loading import get_vector_datasets, load_two_publication_sets
from paths import OUTPUT_DIR
from utils import get_candidate_pairs_between_blocks, convert_matches_to_indices_df, show_tuples_behind_indices_pair

BASELINE_OUTPUT = f"{OUTPUT_DIR}/baseline_cosine.csv"
DEDUPLICATED_ACM_OUTPUT = f"{OUTPUT_DIR}/deduplicated_acm.csv"
DEDUPLICATED_DBLP_OUTPUT = f"{OUTPUT_DIR}/deduplicated_dblp.csv"
ER_PIPELINE_NGRAM_COSINE_OUTPUT = f"{OUTPUT_DIR}/Matched_Entities_Ngram_Cosine_Indices.csv"


# FIRST PIPE FOR ENTITY RESOLUTION
def er_ngram_cosine_pipe(n=2):
    df_acm, df_dblp = load_two_publication_sets()
    print(df_dblp.columns)

    # Vectorization using TF-IDF
    vector_space1, vector_space2 = get_vector_datasets(df_acm, df_dblp)

    blocks1 = create_ngram_word_blocks(df_acm, "Combined_acm", n)
    blocks2 = create_ngram_word_blocks(df_dblp, "Combined_dblp", n)
    candidate_pairs_set = get_candidate_pairs_between_blocks(blocks1, blocks2)

    # Set a similarity threshold
    threshold = 0.8
    
    matching_pairs = []
    for idx1,idx2 in candidate_pairs_set: 
        sim = cosine_similarity(vector_space1[idx1].reshape(1, -1), vector_space2[idx2].reshape(1, -1))[0, 0]
        #print(f"Cosine Similarity between df_acm[{idx1}] and df_dblp[{idx2}]: {similarity}")
        if sim > threshold:
            matching_pairs.append((idx1,idx2))
    columns = ["idx1","idx2"]
    df_matches = pd.DataFrame(matching_pairs, columns=columns)
    df_matches.to_csv(ER_PIPELINE_NGRAM_COSINE_OUTPUT, index=False)
    show_tuples_behind_indices_pair(ER_PIPELINE_NGRAM_COSINE_OUTPUT, "truetuples.csv")

    # get indices from candidate pairs
    indices_pairs_df = convert_matches_to_indices_df(df_acm, df_dblp, df_matches)
    
    acm_clus, dblp_clus = assign_to_clusters(indices_df=indices_pairs_df)

    # use this function to write a file containing the papers belonging to each cluster
    cluster_check(acm_clus, dblp_clus, df_acm, df_dblp)

    representatives_dict = create_cluster_representatives_dict(dblp_clus, df_dblp)

    create_deduplicated_dataset(df_acm, acm_clus, representatives_dict, DEDUPLICATED_ACM_OUTPUT)
    create_deduplicated_dataset(df_dblp, dblp_clus, representatives_dict, DEDUPLICATED_DBLP_OUTPUT)

# BASELINE
def create_baseline(threshold):
    df1, df2 = load_two_publication_sets()

    # Vectorization using TF-IDF
    vector_space1, vector_space2 = get_vector_datasets(df1, df2)
    
    # calculate similarities for all pairs
    cosine_sim = cosine_similarity(vector_space1, vector_space2)
    # Find matching pairs
    matching_pairs = []
    for i in range(len(cosine_sim)):
        for j in range(len(cosine_sim[i])):
            if cosine_sim[i, j] > threshold:
                #print(f"Match: {(df1.iloc[i]['Combined'], df2.iloc[j]['Combined'])} ")
                matching_pairs.append((i, j))
    columns = ["idx1","idx2"]
    df_matches = pd.DataFrame(matching_pairs, columns=columns)
    df_matches.to_csv(BASELINE_OUTPUT, index=False)
    #show_tuples_behind_indices_pair("indices.csv", "truetuples.csv")

def evaluate(baseline_file, matches_file):
    baseline_df = pd.read_csv(baseline_file)
    matches_df = pd.read_csv(matches_file)

    # Count the number of true positives
    true_positives = pd.merge(matches_df, baseline_df, on=['idx1', 'idx2']).shape[0]

    # Count the number of false positives
    false_positives = matches_df.shape[0] - true_positives

    # Count the number of false negatives
    false_negatives = baseline_df.shape[0] - true_positives

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def assign_to_clusters(indices_df):
    acm_clusters = {}
    dblp_clusters = {}
    next_cluster_idx = 0
    
    for _, row in indices_df.iterrows():
        idx1, idx2 = row['idx1'], row['idx2']
        
        if idx1 in acm_clusters and idx2 in dblp_clusters:
            continue  
            
        if idx1 not in acm_clusters and idx2 not in dblp_clusters:
            acm_clusters[idx1] = next_cluster_idx
            dblp_clusters[idx2] = next_cluster_idx
            
            # get the other matches of the dblp dataset to the idx1 and add them to the cluster
            for _, sub_row in indices_df[indices_df['idx1'] == idx1].iterrows():
                if sub_row['idx2'] not in dblp_clusters:
                    dblp_clusters[sub_row['idx2']] = next_cluster_idx
            
            # get the other matches of the acm dataset to the idx2 and add them to the cluster
            for _, sub_row in indices_df[indices_df['idx2'] == idx2].iterrows():
                if sub_row['idx1'] not in acm_clusters:
                    acm_clusters[sub_row['idx1']] = next_cluster_idx
            
            next_cluster_idx += 1
        elif idx1 in acm_clusters and idx2 not in dblp_clusters:
            current_cluster_idx = acm_clusters[idx1]
            dblp_clusters[idx2] = current_cluster_idx
        elif idx2 in dblp_clusters and idx1 not in acm_clusters:
            current_cluster_idx = dblp_clusters[idx2]
            acm_clusters[idx1] = current_cluster_idx
    
    return acm_clusters, dblp_clusters

def cluster_check(dict1, dict2, df1, df2):
    merged_dict = {}

    # Add unique values and initialize empty lists for dict1
    for key, value in dict1.items():
        row = df1[df1['paperId_acm'] == key]
        title_value = row['title_acm'].values[0] if not row.empty else None
        if title_value is not None:
            merged_dict.setdefault(value, []).append(title_value)

    # Add unique values and initialize empty lists for dict2
    for key, value in dict2.items():
        row = df2[df2['paperId_dblp'] == key]
        title_value = row['title_dblp'].values[0] if not row.empty else None
        if title_value is not None:
            merged_dict.setdefault(value, []).append(title_value)

    with open(f"{OUTPUT_DIR}/output.txt", 'w') as file:
        for key, val in merged_dict.items():
            print(f"Cluster ID: {key}", file=file)
            print(f"Papers: {val}", file=file)
            print('', file=file)

    #return merged_dict

def create_cluster_representatives_dict(clusters, df):
    
    cluster_representatives = {}

    id_column = [col for col in df.columns if col.startswith("paperId")][0]
    
    for key, value in clusters.items():
        if value not in cluster_representatives:
            matching_row = df[df[id_column] == key].iloc[0].to_numpy()
            cluster_representatives[value] = matching_row
    
    return cluster_representatives

def create_deduplicated_dataset(original_df, clusters, representatives, file_path):
    deduplicated_df = original_df.copy()

    id_column = [col for col in deduplicated_df.columns if col.startswith("paperId")][0]

    # go through cluster to find the elements that were matched and assigned to a cluster number
    # these elements are then replaced by the representative of the cluster
    for key, value in clusters.items():
        matching_row_idx = deduplicated_df[deduplicated_df[id_column] == key].index
        representative_value = representatives[value]
        # use iloc to set values for individual cells using column indices
        for col_index in range(len(deduplicated_df.columns)):
            deduplicated_df.iloc[matching_row_idx, col_index] = representative_value[col_index]
    
    # drop the combined column
    deduplicated_df = deduplicated_df.drop(deduplicated_df.columns[-1], axis=1)
    deduplicated_df = deduplicated_df.drop_duplicates()
    
    deduplicated_df.to_csv(file_path, index=False)
    #return deduplicated_df



def main():
    # BASELINE FILE CREATION
    start_time = time.time()
    create_baseline(threshold=0.8)
    print(f"Elapsed time for baseline creation: {time.time() - start_time}") 
    
    # BLOCKING AND MATCHING 
    start_time = time.time()
    er_ngram_cosine_pipe(n=4)
    print(f"Elapsed time for matching: {time.time() - start_time}") 

    # EVALUATION: get precision, recall and f-measure 
    precision, recall, f1_score = evaluate(BASELINE_OUTPUT, ER_PIPELINE_NGRAM_COSINE_OUTPUT)
    # Print the results
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()