import time
from blocking import blocking
from data_loading import load_two_publication_sets
from er_block_match import *
from matching import baseline_matching, matching


def evaluate(df, bs_df, threshold, match='jaccard'):
    bs_df_match = bs_df[bs_df.similarity > threshold].sort_values(by=['similarity'],
                                                                        ascending=False)
    df_match = df[df.similarity > threshold].sort_values(by=['similarity'],
                                                                    ascending=False)
    f1, prec, rec = f1_evaluation(df_match, bs_df_match)
    
    return f1, prec, rec
            
def entity_resolution_experiments():
    df_acm, df_dblp = load_two_publication_sets()
        
    experiment_configs = [
        {
            "blocking": 'ngram_word_blocks',
            "matching": 'cosine',
            "blocking_params": {"n": 4},
        },
        {
            "blocking": 'st',
            "matching": 'jaccard',
            "blocking_params": {},
            "matching_weights": [0.7, 0.3, 0]
        },
        {
            'blocking': 'st',
            'matching': 'trigram',
            "blocking_params": {},
            'matching_weights': [0.7, 0.3, 0]
        },
        {
            'blocking': 'st',
            'matching': 'jaccard',
            "blocking_params": {},
            'matching_weights': [0.8, 0.2, 0]
        },
        {
            'blocking': 'token',
            'matching': 'jaccard',
            "blocking_params": {},
            'matching_weights': [0.5, 0.2, 0.3]
        },
        {
            'blocking': 'token',
            'matching': 'trigram',
            "blocking_params": {},
            'matching_weights': [0.5, 0.2, 0.3]
        }, 
        {
            'blocking': 'token',
            'matching': 'jaccard',
            "blocking_params": {},
            'matching_weights': [0.5, 0.3, 0.2]
        }
    ]
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    start_timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open("output.txt", "a") as f:
        for i, config in enumerate(experiment_configs):
            print(f"#{i}: {config['blocking']}, {config['matching']}")
            start= time.time()
            blocks = blocking(df_acm, df_dblp, blocking_scheme=config['blocking'], params=config['blocking_params'])
            df_block_pairs = pd.concat([df_acm.loc[blocks[:, 0]].reset_index(),
                             df_dblp.loc[blocks[:, 1]].reset_index()], axis=1)
            
            df_pairs = matching(df_acm, df_dblp, df_block_pairs, config['matching'], weights=config.get('matching_weights'))

            end = time.time()
            print(f'Time needed for blocking and matching: {end-start}')

            start = time.time()
            bs_df_pairs = baseline_matching(df_acm, df_dblp, match=config['matching'], weights=config.get('matching_weights'))
            end = time.time()
            print(f'Time needed for baseline creation: {end-start}')

            for threshold in thresholds: 
                f1, prec, rec = evaluate(df_pairs, bs_df_pairs, threshold, match=config['matching'])
                print(f'threshold: {threshold}, f1: {f1}, precision: {prec}, recall: {rec}')


if __name__ == "__main__":
    entity_resolution_experiments()