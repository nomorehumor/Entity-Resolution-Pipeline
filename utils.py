import csv
import pandas as pd
from data_loading import load_two_publication_sets
import numpy as np

def get_symbol_ngrams(text, number=3):
    if not text:
        return set()
    text = ' ' * (number - 1) + text + ' ' * (number - 1)
    ngrams = set()
    for x in range(0, len(text) - number + 1):
        ngrams.add(text[x:x + number])
    return ngrams

# Function to create n-gram blocks for a dataframe
def get_words_ngrams(text, n):
    words = text.split()

    # list containing tuples with n subsequent words
    ngrams_list = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    # list containing strings (which are basically n subsequent words)
    return [' '.join(gram) for gram in ngrams_list]  

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

            real_pairs.append((df1.iloc[idx1]["Combined_acm"],df2.iloc[idx2]["Combined_dblp"]))
        columns = ["Combined_acm", "Combined_dblp"]
        df_matches = pd.DataFrame(real_pairs, columns=columns)
        df_matches.to_csv(newfilename, encoding='utf-8-sig', index=False)


def get_candidate_pairs_between_blocks(blocks1, blocks2):
    """
    Helper function to create candidate pairs between blocks
    """
    candidate_pairs = set()
    for ngram in blocks1:
        if ngram in blocks2:
            pairs = [(id1, id2) for id1 in blocks1[ngram] for id2 in blocks2[ngram]]
            candidate_pairs.update(pairs) # add all candidate pairs to set
    return np.array(list(candidate_pairs))

# instead of storing the rownumbers in the dataframes, get the actual PaperID in the dataframes
def convert_matches_to_indices_df(df_acm, df_dblp, df_matches):
    indices = [(df_acm.iloc[row["idx1"]]['paperId_acm'], df_dblp.iloc[row["idx2"]]['paperId_dblp']) for _, row in df_matches.iterrows()]
    indices_df = pd.DataFrame(indices, columns=['idx1', 'idx2'])
    return indices_df


def create_cartesian_product_baseline(df_acm, df_dblp):
    # all_pairs = np.array([list(pair) for pair in itertools.product(df_acm.index, df_dblp.index)])
    # bs_df = pd.concat([df_acm.loc[all_pairs[:, 0]].reset_index(),
    #                           df_dblp.loc[all_pairs[:, 1]].reset_index()], axis=1)
    
    df_acm = df_acm.reset_index().rename(columns={'index': 'index_acm'})
    df_dblp = df_dblp.reset_index().rename(columns={'index': 'index_dblp'})
    df_acm['key'] = 1
    df_dblp['key'] = 1      
    bs_df = pd.merge(df_acm, df_dblp, on='key').drop('key', axis=1)
    return bs_df