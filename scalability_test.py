import random

import pandas as pd

from distributed_er_pipeline import distributed_er_pipeline
from paths import ACM_DATASET_FILE, DBLP_DATASET_FILE


def dataset_replication():
    num = 10
    column_names = ['paperId', 'title', 'authors', 'venue', 'year']
    dtype = {'PaperID': str, 'Title': str, 'Authors': str, 'Venue': str, 'Year': int}
    df_acm = pd.read_csv(ACM_DATASET_FILE, sep='|', skiprows=1, names=column_names, encoding='utf-8-sig',
                         dtype=dtype)
    df_dblp = pd.read_csv(DBLP_DATASET_FILE, sep='|', skiprows=1, names=column_names, encoding='utf-8-sig',
                          dtype=dtype)

    for replication_factor in range(1, num + 1):
        replicated_acm = pd.concat([df_acm] * replication_factor, ignore_index=True)
        replicated_dblp = pd.concat([df_dblp] * replication_factor, ignore_index=True)

        for dataset in [replicated_acm, replicated_dblp]:
            dataset.authors = dataset.authors.apply(lambda x: (x.upper() if isinstance(x, str) else '') + ' ' + str(random.randint(1, 100000)))
            dataset.title = dataset.title.apply(lambda x: (x.upper() if isinstance(x, str) else '') + ' ' + str(random.randint(1, 100000)))
            dataset.paperId = dataset.paperId.apply(
                lambda x: str(x) + str(random.randint(1, 100000)))

        replicated_acm.to_csv(f'data/replicated_acm_{replication_factor}.csv', sep='|', encoding='utf-8-sig',
                              index=False)
        replicated_dblp.to_csv(f'data/replicated_dblp_{replication_factor}.csv', sep='|', encoding='utf-8-sig',
                               index=False)


def check_execution_time():
    num = 10
    exec_time = []
    for i in range(num):
        acm_file = f'data/replicated_acm_{i + 1}.csv'
        dblp_file = f'data/replicated_dblp_{i + 1}.csv'
        time = distributed_er_pipeline(acm_file, dblp_file, save=False)
        exec_time.append(time)
    print(exec_time)
