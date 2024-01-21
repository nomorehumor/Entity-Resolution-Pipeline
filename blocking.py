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


 
def get_candidate_pairs_between_blocks(blocks1, blocks2):
    """
    Helper function to create candidate pairs between blocks
    """
    candidate_pairs = set()
    for ngram in blocks1:
        if ngram in blocks2:
            pairs = [(id1, id2) for id1 in blocks1[ngram] for id2 in blocks2[ngram]]
            candidate_pairs.update(pairs) # add all candidate pairs to set
    return candidate_pairs