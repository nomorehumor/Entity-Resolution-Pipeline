from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from paths import ACM_DATASET_FILE, DBLP_DATASET_FILE


# LOADING DATA 
def read_file(filename, dataset_origin):
    column_names = ['paperId', 'title', 'authors', 'venue', 'year']
    df = pd.read_csv(filename, sep='|', skiprows=1, names=column_names, encoding='utf-8-sig', dtype={'PaperID': str, 'Title': str, 'Authors': str, 'Venue': str, 'Year': int})
    preprocessing(df)

    df.drop_duplicates(subset=["paperId"], keep='first', ignore_index=True, inplace=True)
    df.fillna("", inplace=True)

    column_names = [f"{column_name}_{dataset_origin}" for column_name in df.columns]
    df.columns = column_names
    df.index.names = [f'index_{dataset_origin}']
    # df.dropna(ignore_index=True, inplace=True)
    # df.drop_duplicates(keep='first', ignore_index=True, inplace=True)
    return df


def preprocessing(df):
    df['title'] = (df['title'].str.lower()
                   .replace("[^a-z0-9]", " ", regex=True)
                   .replace(" +", " ", regex=True)
                   .str.strip())
    df['authors'] = (df['authors']
                     .str.lower()
                     .replace("[^a-z0-9]", " ", regex=True)
                     .replace(" +", " ", regex=True)
                     .str.strip())

    df['venue'] = (df['venue']
                   .str.lower()
                   .replace(" +", " ", regex=True)
                   .str.strip())

    combine_attributes = lambda row: f"{row['title']} {row['authors']} {row['year']}"
    df["Combined"] = df.apply(combine_attributes, axis=1)


def load_two_publication_sets():
    df_acm = read_file(ACM_DATASET_FILE, "acm")
    df_dblp = read_file(DBLP_DATASET_FILE, "dblp")

    return df_acm, df_dblp


def get_vector_datasets(df_acm, df_dblp):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df_acm["Combined_acm"].values.tolist() + df_dblp["Combined_dblp"].values.tolist())
    vector_space1 = vectorizer.transform(df_acm["Combined_acm"]).toarray()
    vector_space2 = vectorizer.transform(df_dblp["Combined_dblp"]).toarray()
    return vector_space1, vector_space2
