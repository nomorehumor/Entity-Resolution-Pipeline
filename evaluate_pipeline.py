import csv
import os
import time

import pandas as pd

from pipeline.clustering import connected_components, deduplicate_datasets
from pipeline.blocking import blocking
from pipeline.data_loading import load_two_publication_sets
from pipeline.matching import baseline_matching, matching
from paths import OUTPUT_DIR


def evaluate(df, bs_df, threshold):
    bs_df_match = bs_df[bs_df.similarity > threshold].sort_values(by=['similarity'],
                                                                        ascending=False)
    df_match = df[df.similarity > threshold].sort_values(by=['similarity'],
                                                                    ascending=False)
    f1, prec, rec = f1_evaluation(df_match, bs_df_match)

    return f1, prec, rec


def f1_evaluation(df, bs_df):
    tp = len(pd.merge(df, bs_df, how='inner', on=['index_acm', 'index_dblp']))
    fp = len(df) - tp
    fn = len(bs_df) - tp
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return f1, precision, recall

def entity_resolution_experiments():
    df_acm, df_dblp = load_two_publication_sets()

    experiment_configs = [
        {
            "blocking": 'ngram_word_blocks',
            "matching": 'cosine',
            "blocking_params": {"n": 4},
        },
        {
            "blocking": 'ngram_word_blocks',
            "matching": 'cosine',
            "blocking_params": {"n": 3},
        },
        {
            "blocking": 'ngram_word_blocks',
            "matching": 'jaccard',
            "blocking_params": {"n": 3},
            "matching_weights": [0.3, 0.3, 0.3]
        },
        {
            "blocking": 'ngram_word_blocks',
            "matching": 'trigram',
            "blocking_params": {"n": 3},
            "matching_weights": [0.3, 0.3, 0.3]
        },
        {
            "blocking": 'st',
            "matching": 'cosine',
            "blocking_params": {},
        },
        {
            "blocking": 'token',
            "matching": 'cosine',
            "blocking_params": {},
        },
        {
            "blocking": 'st',
            "matching": 'jaccard',
            "blocking_params": {},
            "matching_weights": [0.7, 0.3, 0]
        },
        {
            'blocking': 'token',
            'matching': 'jaccard',
            "blocking_params": {},
            'matching_weights': [0.7, 0.3, 0]
        },
        {
            'blocking': 'st',
            'matching': 'jaccard',
            "blocking_params": {},
            'matching_weights': [0.3, 0.3, 0.3]
        },
        {
            'blocking': 'token',
            'matching': 'jaccard',
            "blocking_params": {},
            'matching_weights': [0.3, 0.3, 0.3]
        },
        {
            'blocking': 'st',
            'matching': 'jaccard',
            "blocking_params": {},
            'matching_weights': [0, 0.7, 0.3]
        },
        {
            'blocking': 'token',
            'matching': 'jaccard',
            "blocking_params": {},
            'matching_weights': [0, 0.7, 0.3]
        },
        {
            "blocking": 'st',
            "matching": 'trigram',
            "blocking_params": {},
            "matching_weights": [0.7, 0.3, 0]
        },
        {
            'blocking': 'token',
            'matching': 'trigram',
            "blocking_params": {},
            'matching_weights': [0.7, 0.3, 0]
        },
        {
            'blocking': 'st',
            'matching': 'trigram',
            "blocking_params": {},
            'matching_weights': [0.3, 0.3, 0.3]
        },
        {
            'blocking': 'token',
            'matching': 'trigram',
            "blocking_params": {},
            'matching_weights': [0.3, 0.3, 0.3]
        },
        {
            'blocking': 'st',
            'matching': 'trigram',
            "blocking_params": {},
            'matching_weights': [0, 0.7, 0.3]
        },
        {
            'blocking': 'token',
            'matching': 'trigram',
            "blocking_params": {},
            'matching_weights': [0, 0.7, 0.3]
        },
        {
            "blocking": 'st',
            "matching": 'levenshtein',
            "blocking_params": {},
            "matching_weights": [0.7, 0.3, 0]
        },
        {
            'blocking': 'token',
            'matching': 'levenshtein',
            "blocking_params": {},
            'matching_weights': [0.7, 0.3, 0]
        },
        {
            'blocking': 'st',
            'matching': 'levenshtein',
            "blocking_params": {},
            'matching_weights': [0.3, 0.3, 0.3]
        },
        {
            'blocking': 'token',
            'matching': 'levenshtein',
            "blocking_params": {},
            'matching_weights': [0.3, 0.3, 0.3]
        },
        {
            'blocking': 'st',
            'matching': 'levenshtein',
            "blocking_params": {},
            'matching_weights': [0, 0.7, 0.3]
        },
        {
            'blocking': 'token',
            'matching': 'levenshtein',
            "blocking_params": {},
            'matching_weights': [0, 0.7, 0.3]
        },
        {
            'blocking': 'ngram_word_blocks',
            'matching': 'levenshtein',
            "blocking_params": {"n": 3},
            'matching_weights': [0.3, 0.3, 0.3]
        },
        {
            'blocking': 'ngram_word_blocks',
            'matching': 'levenshtein',
            "blocking_params": {"n": 3},
            'matching_weights': [0, 0.7, 0.3]
        },
        {
            'blocking': 'ngram_word_blocks',
            'matching': 'levenshtein',
            "blocking_params": {"n": 3},
            'matching_weights': [0, 0.7, 0.3]
        }
    ]

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    chosen_threshold = 0.8
    start_timestamp = time.strftime("%Y%m%d-%H%M%S")
    for i, config in enumerate(experiment_configs):
        print(f"#{i}: {config['blocking']}, {config['matching']}")
        pipeline_start = time.time()
        pairs = blocking(df_acm, df_dblp, blocking_scheme=config['blocking'], params=config['blocking_params'])
        df_pairs = matching(df_acm, df_dblp, pairs, config['matching'], weights=config.get('matching_weights'))

        pipeline_end = time.time()
        print(f'Time needed for blocking and matching: {pipeline_end - pipeline_start}')

        bs_start = time.time()
        bs_df_pairs = baseline_matching(df_acm, df_dblp, matching_function=config['matching'],
                                        weights=config.get('matching_weights'))
        bs_matching_end = time.time()
        print(f'Time needed for baseline creation & matching : {bs_matching_end - bs_start}')

        for threshold in thresholds:
            f1, prec, rec = evaluate(df_pairs, bs_df_pairs, threshold)
            print(f'threshold: {threshold}, f1: {f1}, precision: {prec}, recall: {rec}')
            save_result(config, start_timestamp, threshold, f1, prec, rec, pipeline_end - pipeline_start)

        clusters = connected_components(df_pairs[df_pairs.similarity > chosen_threshold])
        df_acm_dedup, df_dblp_dedup = deduplicate_datasets(df_acm, df_dblp, clusters)
        df_acm_dedup.to_csv(f'{OUTPUT_DIR}/ACM_deduplicated_{i}.csv')
        df_dblp_dedup.to_csv(f'{OUTPUT_DIR}/DBLP_deduplicated_{i}.csv')


def save_result(config, timestamp, threshold, f1, prec, rec, pipeline_execution_time):
    filename = f"{OUTPUT_DIR}/result_{timestamp}.csv"
    file_exists = os.path.exists(filename)
    with open(filename, "a") as f:
        writer = csv.writer(f, delimiter="|")
        if not file_exists:
            writer.writerow(["config", "threshold", "f1", "precision", "recall", "execution_time"])
        writer.writerow([config, threshold, f1, prec, rec, pipeline_execution_time])


if __name__ == "__main__":
    entity_resolution_experiments()
