import csv
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

            real_pairs.append((df1.iloc[idx1]["Combined"],df2.iloc[idx2]["Combined"]))
        columns = ["Combined_1", "Combined_2"]
        df_matches = pd.DataFrame(real_pairs, columns=columns)
        df_matches.to_csv(newfilename, encoding='utf-8-sig', index=False)