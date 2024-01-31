from paths import OUTPUT_DIR


def deduplicate_datasets(df_acm, df_dblp, clusters, config_num=0):
    idx_acm, idx_dblp = [], []

    for key in clusters.keys():
        if len(clusters[key]) > 2:
            id_acm = [int(el[2:]) for el in clusters[key] if el.startswith('1')]
            idx_acm.extend(id_acm[1:])
            id_dblp = [int(el[2:]) for el in clusters[key] if el.startswith('2')]
            idx_dblp.extend(id_dblp[1:])
    
    if any("Combined" in col for col in df_acm.columns):
        df_acm = df_acm.drop(columns=[col for col in df_acm.columns if "Combined" in col])
    if any("Combined" in col for col in df_dblp.columns):
        df_dblp = df_dblp.drop(columns=[col for col in df_dblp.columns if "Combined" in col])
        
    df_acm.drop(idx_acm).to_csv(f'{OUTPUT_DIR}/ACM_deduplicated_{config_num}.csv')
    df_dblp.drop(idx_dblp).to_csv(f'{OUTPUT_DIR}/DBLP_deduplicated_{config_num}.csv')

