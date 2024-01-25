import csv
from collections import defaultdict

import pandas as pd
from nltk import word_tokenize

from data_loading import load_two_publication_sets


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

            real_pairs.append((df1.iloc[idx1]["Combined"], df2.iloc[idx2]["Combined"]))
        columns = ["Combined_1", "Combined_2"]
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
            candidate_pairs.update(pairs)  # add all candidate pairs to set
    return candidate_pairs


def get_token_blocks(df, stop_words):
    blocks = defaultdict(list)

    for idx, row in enumerate(df.itertuples()):
        string = " ".join([str(value) for value in row if not pd.isna(value)])
        tokens = set(
            [word for word in word_tokenize(string) if word not in stop_words]
        )
        for token in tokens:
            blocks[token].append(idx)

    blocks = {
        key: indices
        for key, indices in blocks.items()
        if 1000 > len(indices) > 1
    }

    return blocks