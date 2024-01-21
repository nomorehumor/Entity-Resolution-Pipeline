from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from paths import ACM_DATASET_FILE, DBLP_DATASET_FILE


# LOADING DATA 
def read_file(filename):
    column_names = ['PaperID', 'Title', 'Authors', 'Venue', 'Year']
    df = pd.read_csv(filename, sep='|', names=column_names, skiprows=1, encoding='utf-8-sig', dtype={'PaperID': str, 'Title': str, 'Authors': str, 'Venue': str, 'Year': int})
    return df

def load_two_publication_sets():
    df1 = read_file(DBLP_DATASET_FILE)
    df2 = read_file(ACM_DATASET_FILE)

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