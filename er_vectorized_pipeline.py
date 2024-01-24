import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

from blocking import create_ngram_word_blocks
from data_loading import get_vector_datasets, load_two_publication_sets
from paths import OUTPUT_DIR
from utils import get_candidate_pairs_between_blocks

BASELINE_OUTPUT = f"{OUTPUT_DIR}/baseline_cosine.csv"
ER_PIPELINE_NGRAM_COSINE_OUTPUT = f"{OUTPUT_DIR}/Matched_Entities_Ngram_Cosine_Indices.csv"


# FIRST PIPE FOR ENTITY RESOLUTION
def er_ngram_cosine_pipe(n=2):
    df1, df2 = load_two_publication_sets()

    # Vectorization using TF-IDF
    vector_space1, vector_space2 = get_vector_datasets(df1, df2)

    blocks1 = create_ngram_word_blocks(df1, "Combined_dblp", n)
    blocks2 = create_ngram_word_blocks(df2, "Combined_acm", n)
    candidate_pairs_set = get_candidate_pairs_between_blocks(blocks1, blocks2)

    # Set a similarity threshold
    threshold = 0.8

    matching_pairs = []
    for idx1,idx2 in candidate_pairs_set: 
        sim = cosine_similarity(vector_space1[idx1].reshape(1, -1), vector_space2[idx2].reshape(1, -1))[0, 0]
        #print(f"Cosine Similarity between df1[{idx1}] and df2[{idx2}]: {similarity}")
        if sim > threshold:
            matching_pairs.append((idx1,idx2))
    columns = ["idx1","idx2"]
    df_matches = pd.DataFrame(matching_pairs, columns=columns)
    df_matches.to_csv(ER_PIPELINE_NGRAM_COSINE_OUTPUT, index=False)
    # show_tuples_behind_indices_pair("Matched_Entities_Ngram_Cosine_Indices.csv", "truetuples.csv")

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