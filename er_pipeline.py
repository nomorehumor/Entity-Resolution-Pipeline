import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import csv

# LOADING DATA 
def read_file(filename):
    column_names = ['PaperID', 'Title', 'Authors', 'Venue', 'Year']
    df = pd.read_csv(filename, sep='|', names=column_names, skiprows=1, encoding='utf-8-sig', dtype={'PaperID': str, 'Title': str, 'Authors': str, 'Venue': str, 'Year': int})
    return df

def load_two_publication_sets():
    df1 = read_file("DBLP_1995_2004.csv")
    df2 = read_file("ACM_1995_2004.csv")

    # Combine relevant attributes into a single string
    combine_attributes = lambda row: f"{row['Title']} {row['Authors']} {row['Year']}"
    df1["Combined"] = df1.apply(combine_attributes, axis=1)
    df2["Combined"] = df2.apply(combine_attributes, axis=1)
    return df1, df2

def get_vector_datasets(df1, df2):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df1["Combined"].values.tolist() + df2["Combined"].values.tolist())
    vector_space1 = vectorizer.transform(df1["Combined"]).toarray()
    vector_space2 = vectorizer.transform(df2["Combined"]).toarray()
    return vector_space1, vector_space2

# FIRST BLOCKING SCHEME
def extract_ngrams(text, n):
    words = text.split()

    # list containing tuples with n subsequent words
    ngrams_list = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    # list containing strings (which are basically n subsequent words)
    return [' '.join(gram) for gram in ngrams_list]  

# Function to create n-gram blocks for a dataframe
def create_ngram_blocks(df, column, n):
    blocks = {}
    for idx, row in df.iterrows():
        ngrams_list = extract_ngrams(row[column], n)
        for ngram in ngrams_list:
            if ngram not in blocks:
                blocks[ngram] = []
            blocks[ngram].append(idx) # only append idx of row in dataframe to save more space
    return blocks

# HELPER FUNCTION TO CREATE CANDIDATE PAIRS BETWEEN BLOCKS
def get_candidate_pairs_between_blocks(blocks1, blocks2):
    candidate_pairs = set()
    for ngram in blocks1:
        if ngram in blocks2:
            pairs = [(id1, id2) for id1 in blocks1[ngram] for id2 in blocks2[ngram]]
            candidate_pairs.update(pairs) # add all candidate pairs to set
    return candidate_pairs


# HELPER TO SEE WHETHER MATCHED INDICES ARE CORRECT
def show_tuples_behind_indices_pair(filename, newfilename):
    df1, df2 = load_two_publication_sets()

    with open(filename, 'r') as indices_pairs:
        csv_reader = csv.reader(indices_pairs)

        # Skip the header if it exists
        next(csv_reader, None)

        real_pairs = []
        for row in csv_reader:
            idx1 = int(row[0])
            idx2 = int(row[1])

            real_pairs.append((df1.iloc[idx1]["Combined"],df2.iloc[idx2]["Combined"]))
        columns = ["Combined_1", "Combined_2"]
        df_matches = pd.DataFrame(real_pairs, columns=columns)
        df_matches.to_csv(newfilename, encoding='utf-8-sig', index=False)


# FIRST PIPE FOR ENTITY RESOLUTION
def er_ngram_cosine_pipe(n=2):
    df1, df2 = load_two_publication_sets()

    # Vectorization using TF-IDF
    vector_space1, vector_space2 = get_vector_datasets(df1, df2)

    blocks1 = create_ngram_blocks(df1, "Combined", n)
    blocks2 = create_ngram_blocks(df2, "Combined", n)
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
    df_matches.to_csv("Matched_Entities_Ngram_Cosine_Indices.csv", index=False)
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
    df_matches.to_csv("baseline_cosine.csv", index=False)
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

    # Print the results
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

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
    evaluate("baseline_cosine.csv", "Matched_Entities_Ngram_Cosine_Indices.csv")

if __name__ == "__main__":
    main()