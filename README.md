# Entity Resolution (ER) Pipeline

This repository contains implementation of an ER pipeline.

Directories: 
```
- data # extracted datasets
- output # experiment results, Matched_Entities.csv, deduplicated datasets
- pipeline # code for different stages of ER pipeline
- scripts # 2 other approaches that weren't fully included to the main script
```

## 0. Clone Repo & Install Dependencies

```
git clone
cd dia-entity-resolution
pip install -r requirements.txt
```

## 1. Data Acquisition and Preparation

Download datasets:
```
wget https://lfs.aminer.cn/lab-datasets/citation/dblp.v8.tgz
wget https://lfs.aminer.cn/lab-datasets/citation/citation-acm-v8.txt.tgz
```

Run the extraction script:

```
python3 extract_to_csv.py
```

Files will be stored in `data/` folder

## 2. Run Pipeline

To execute the experiments:

```
python3 evaluate_pipeline.py
```

To run the pipeline (n-gram blocking + Levenshtein):

```
python3 run_pipeline.py
```

To run distributed pipeline:

```
python3 distributed_er_pipeline.py
```

## 3. Alternative Approach

We also have an alternative approach with blocking and matching using n-gram words and cosine function. It also uses different clustering algorithm.
To run it:

```
python3 scripts/er_vectorized_pipeline.py
```