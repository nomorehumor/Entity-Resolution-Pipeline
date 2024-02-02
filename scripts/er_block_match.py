import itertools
import os
import re
import string
import time

import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import numpy as np
import pandas as pd
from pipeline.blocking import blocking_by_year, token_blocking
from pipeline.data_loading import preprocessing
from pipeline.matching import string_matching

from paths import ACM_DATASET_FILE, DBLP_DATASET_FILE, OUTPUT_DIR

def data_preparation():
    df1 = pd.read_csv(ACM_DATASET_FILE, sep='|')
    df2 = pd.read_csv(DBLP_DATASET_FILE, sep='|')
    df1.dropna(ignore_index=True, inplace=True)
    df2.dropna(ignore_index=True, inplace=True)
    preprocessing(df1)
    df1.index.names = ['index_acm']
    df1.columns = ['paperID_acm', 'title_acm', 'authors_acm', 'venue_acm', 'year_acm']
    preprocessing(df2)
    df2.columns = ['paperID_dblp', 'title_dblp', 'authors_dblp', 'venue_dblp', 'year_dblp']
    df2.index.names = ['index_dblp']

    return df1, df2

def evaluate(df, bs_df, f, match='jaccard'):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for threshold in thresholds:
        if match=='trigram':
            bs_df_trigram = bs_df[bs_df.trigram_sim > threshold].sort_values(by=['trigram_sim'],
                                                                             ascending=False)
            match_df_trigram = df[df.trigram_sim > threshold].sort_values(by=['trigram_sim'],
                                                                          ascending=False)
            f1, prec, rec = f1_evaluation(match_df_trigram, bs_df_trigram)
            print(f'Trigram: threshold: {threshold}, f1: {f1}, precision: {prec}, recall: {rec}', file=f)
        else:
            bs_df_jaccard = bs_df[bs_df.jaccard_sim > threshold].sort_values(by=['jaccard_sim'],
                                                                             ascending=False)
            match_df_jaccard = df[df.jaccard_sim > threshold].sort_values(by=['jaccard_sim'],
                                                                          ascending=False)
            f1, prec, rec = f1_evaluation(match_df_jaccard, bs_df_jaccard)
#             print(f'Jaccard: threshold: {threshold}, f1: {f1}, precision: {prec}, recall: {rec}', file=f)


def f1_evaluation(df, bs_df):
    tp = len(pd.merge(df, bs_df, how='inner', on=['index_acm', 'index_dblp']))
    fp = len(df) - tp
    fn = len(bs_df) - tp
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def entity_resolution_experiments(save_csv=False):
    df1, df2 = data_preparation()
    matches = ['jaccard', 'trigram', 'jaccard', 'jaccard', 'trigram', 'jaccard']
    blocks = ['st', 'st', 'st', 'token', 'token', 'token']
    weights = [[0.7, 0.3, 0], [0.7, 0.3, 0], [0.8, 0.2, 0], [0.5, 0.2, 0.3], [0.5, 0.2, 0.3], [0.5, 0.3, 0.2]]
    with open("output.txt", "a") as f:
        for i in range(len(matches)):
            print(f'{blocks[i]}, {matches[i]}, {weights[i]}')
            start= time.time()
            df = blocking_and_matching(df1, df2, block=blocks[i], match=matches[i], weights=weights[i])
            print(f'Time needed for blocking and matching: {time.time() - start}')

            start = time.time()
            bs_df = baseline_matching(df1, df2, match=matches[i], weights=weights[i])
            print(f'Time needed for baseline creation: {time.time() - start}')

            evaluate(df, bs_df, f, match=matches[i])

            if save_csv:
                w = f'{weights[i][0]*10}{weights[i][1]*10}{weights[2][0]*10}'
                df.to_csv(f'{OUTPUT_DIR}/{matches[i]}_{blocks[i]}_{w}.csv')


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    entity_resolution_experiments()
