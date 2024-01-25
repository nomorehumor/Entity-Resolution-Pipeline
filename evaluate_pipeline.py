import time
from blocking import blocking
from data_loading import load_two_publication_sets
from er_block_match import *
from matching import baseline_matching, matching


def evaluate(df, bs_df, threshold):
    bs_df_match = bs_df[bs_df.similarity > threshold].sort_values(by=['similarity'],
                                                                  ascending=False)
    df_match = df[df.similarity > threshold].sort_values(by=['similarity'],
                                                         ascending=False)
    f1, prec, rec = f1_evaluation(df_match, bs_df_match)

    return f1, prec, rec


def entity_resolution_experiments():
    df1, df2 = load_two_publication_sets()

    experiment_configs = [
        {
            "blocking": 'by_year',
            "matching": 'jaccard',
            "matching_weights": [0.7, 0.3, 0]
        },
        {
            'blocking': 'by_year',
            'matching': 'trigram',
            'matching_weights': [0.7, 0.3, 0]
        },
        {
            'blocking': 'by_year',
            'matching': 'jaccard',
            'matching_weights': [0.8, 0.2, 0]
        },
        {
            'blocking': 'token',
            'matching': 'jaccard',
            'matching_weights': [0.5, 0.2, 0.3]
        },
        {
            'blocking': 'token',
            'matching': 'trigram',
            'matching_weights': [0.5, 0.2, 0.3]
        },
        {
            'blocking': 'token',
            'matching': 'jaccard',
            'matching_weights': [0.5, 0.3, 0.2]
        }
    ]

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    with open("files/output.txt", "a") as f:
        for i, config in enumerate(experiment_configs):
            print(f"#{i}: {config['blocking']}, {config['matching']}")
            start = time.time()
            blocks = blocking(df1, df2, blocking_scheme=config['blocking'])
            df_blocking = pd.concat([df1.loc[blocks[:, 0]].reset_index(),
                                     df2.loc[blocks[:, 1]].reset_index()], axis=1)

            matching(df1, df2, df_blocking, config['matching'], weights=config.get('weights'))

            print(f'Time needed for blocking and matching: {time.time() - start}')

            start = time.time()
            bs_df = baseline_matching(df1, df2, match=config['matching'], weights=config['matching_weights'])
            print(f'Time needed for baseline creation: {time.time() - start}')

            for threshold in thresholds:
                f1, prec, rec = evaluate(df_blocking, bs_df, threshold, match=config['matching'])
                print(f'threshold: {threshold}, f1: {f1}, precision: {prec}, recall: {rec}')


if __name__ == "__main__":
    entity_resolution_experiments()
