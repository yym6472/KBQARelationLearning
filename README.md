# KBQARelationLearning

Code for our EMNLP 2021 paper - [Large-Scale Relation Learning for Question Answering over Knowledge Bases with Pre-trained Language Models]()

## Requirements

Install the requirements specified in `requirements.txt`:
```
torch==1.4.0
transformers==4.3.3
tqdm==4.58.0
numpy==1.19.2
tensorboardX==2.1
matplotlib==3.3.4
```

## Data preprocessing

### Preparing WebQSP Dataset

We obtain and process WebQSP dataset using the scripts released by [GraftNet](https://github.com/OceanskySun/GraftNet/tree/master/preprocessing). The scripts will firstly download the original WebQSP dataset as well as the entity links from [STAGG](https://raw.githubusercontent.com/scottyih/STAGG), then they compute the relation and question embeddings using Glove embeddings and run the edge-weighted Personal Page Rank (PPR) algorithm to retrieve a subgraph of freebase for each question. The maximum number of retrieved entities is set to 500.

Our preprocessing scripts are in the `graftnet_preprocessing` folder (We modified the original scripts, making them runnable in python3 environment).

### Preprocessing WebQSP for BERT-based KBQA

To solve KBQA task with BERT, we need to preprocessing the WebQSP dataset, making them a question-context matching task. The model is then trained to predict whether the given candidate entity is the answer of the question. The preprocessing includes:

1. To make use of the textual information of entities and relations in KB facts, we need to find the entity name given the freebase mid (e.g. `<fb:m.03rk0>`). We firstly download the [freebase data dumps](https://developers.google.com/freebase/) to `data/freebase/freebase-rdf-latest.gz`, then use the script `data/freebase/generate_mid_to_name_mapping.py` to obtain the mapping `data/freebase/mid2name.json`.
2. Then we use the script `data/webqsp_name_expanded/process.py` to process the WebQSP dataset in the question-context matching form.

### Preparing Dataset for Relation Learning

We firstly download the [WebRED dataset](https://github.com/google-research-datasets/WebRED) to `data/webred/WebRED` folder and download the [FewRel dataset](http://www.zhuhao.me/fewrel/) to `data/webred_fewrel_matching/fewrel/` folder.

For Relation Extraction (RE) task, we use the script `data/webred/preprocess.py` and the processed datasets are in the `data/webred` folder.

For Relation Matching (RM) task, we use the script `data/webred_matching/preprocess.py` and the processed datasets are in the `data/webred_matching` folder. To further make use of the FewRel dataset, we use the script `data/webred_fewrel_matching/fewrel/process.py` to generate the RM dataset that uses both WebRED and FewRel as data sources in the `data/webred_fewrel_matching` folder.

For Relation Reasoning (RR) task, we modified the [script from BERTRL](https://github.com/zhw12/BERTRL/blob/master/load_data.py) to `data/freebase_pretraining_bertrl/load_data_webqsp.py`, and the generated dataset will be in the `data/freebase_pretraining_bertrl/webqsp` folder.

## Training & Evaluating

Run the scripts in `scripts` folder.

## Citation
```
@inproceedings{yan2021large,
  title={Large-Scale Relation Learning for Question Answering over Knowledge Bases with Pre-trained Language Models},
  author={Yan, Yuanmeng and Li, Rumei and Wang, Sirui and Zhang, Hongzhi and Daoguang, Zan and Zhang, Fuzheng and Wu, Wei and Xu, Weiran},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={3653--3660},
  year={2021}
}
```
