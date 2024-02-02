"# dia-entity-resolution" 

Link to Overleaf: https://www.overleaf.com/7685527817vqfdrjwdmpdd#c3140f

0. Clone Repo & Install Dependencies

```
git clone
cd dia-entity-resolution
pip install -r requirements.txt
```

1. Data Acquisition and Preparation

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

2. Run Pipeline

To execute the experiments