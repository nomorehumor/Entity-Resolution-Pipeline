from utils import get_symbol_ngrams


def matching(df, sim='jaccard', weights=[0.33, 0.33, 0.33]):
    if sim == 'jaccard':
        df['similarity'] = df.apply(lambda x: weights[0] * jaccard_sim(x.title_acm, x.title_dblp)
                        + weights[1] * jaccard_sim(x.authors_acm, x.authors_dblp)
                        + weights[2] * int(x.year_acm == x.year_dblp)
                                    , axis=1)
    elif sim == 'trigram':
        df['similarity'] = df.apply(lambda x: weights[0] * trigram_sim(get_symbol_ngrams(x.title_acm), get_symbol_ngrams(x.title_dblp))
                      + weights[1] * trigram_sim(get_symbol_ngrams(x.authors_acm), get_symbol_ngrams(x.authors_dblp))
                      + weights[2] * int(x.year_acm == x.year_dblp)
                                    , axis=1)
    elif sim == 'levenshtein':
        df['similarity'] = df.apply(lambda x: weights[0] * levenshtein_sim(x.title_one, x.title_two)
                                    + weights[1] * levenshtein_sim(x.authors_acm, x.authors_dblp), axis=1)
        

def jaccard_sim(s1, s2):
    s_intersection = set(set(s1.split()).intersection(set(s2.split())))
    s_union = set(s1.split()).union(set(s2.split()))
    return len(s_intersection) / len(s_union)


def trigram_sim(ngram_1, ngram_2):
    return 2 * len(ngram_1.intersection(ngram_2)) / (len(ngram_1) + len(ngram_2))

def levenshtein_dist(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return max(len(s1), len(s2)) # returns the length of the non-empty string
    if s1[0] == s2[0]:
        return levenshtein_dist(s1[1:], s2[1:])
    else:
        return 1 + min(
            levenshtein_dist(s1[1:], s2), # deletion
            levenshtein_dist(s1, s2[1:]), # insertion
            levenshtein_dist(s1[1:], s2[1:]) # substitution
        )

def levenshtein_sim(s1,s2):
    return 1 - levenshtein_dist(s1,s2) / max(len(s1), len(s2))