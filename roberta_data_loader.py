from typing import List

import json
import my_logging as logging
import random
import tqdm
import math
import torch

from transformers import RobertaTokenizer, RobertaTokenizerFast
from collections import defaultdict


class OriginalWebQSPDataLoader(object):
    """
    对应瑛瑶处理的那版数据（原full_noconstrain），路径：./data/webqsp_yingyao
    """
    def __init__(self, file_path: str, split: str, bert_model_name_or_path: str, shuffle: bool = True):
        samples = json.load(open(file_path, "r"))
        logging.info(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
        samples_for_matching = []
        for sample in samples:
            for item in sample["true"] + sample["false"]:
                samples_for_matching.append({
                    "question": item["q"],
                    "paths": "; ".join(item["path"]),
                    "id": item["id"],
                    "label": item["label"],
                    "candi": item["candi"]
                })
        if shuffle:
            random.shuffle(samples_for_matching)
        self.data = samples_for_matching
        self.num_data = len(self.data)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name_or_path)
        
    def shuffle(self):
        random.shuffle(self.data)
        
    def get_batch(self, iteration, batch_size, max_length):
        batch_data = self.data[iteration * batch_size : (iteration + 1) * batch_size]
        
        q_ids = [sample["id"] for sample in batch_data]
        candidates = [sample["candi"] for sample in batch_data]
        labels = torch.tensor([sample["label"] for sample in batch_data]).long()
        
        bert_inputs = self.tokenizer.batch_encode_plus([[sample["question"], sample["paths"]] for sample in batch_data], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = bert_inputs["input_ids"]
        token_type_ids = bert_inputs["token_type_ids"]
        attention_mask = bert_inputs["attention_mask"]
        
        return input_ids, token_type_ids, attention_mask, labels, q_ids, candidates

    
class NameExpandedWebQSPDataLoader(object):
    """
    对应新处理的那版数据，把大部分缺失的实体名补全了（但仍有少数找不到实体名的，表示为[unknown entity]），路径：./data/webqsp_name_expanded
    
    Args
    ----
    file_path: str
        读取的数据文件路径；
    split: str
        对应属于数据集的哪个split
    bert_model_name_or_path: str
        预训练语言模型的路径，用于加载tokenizer
    shuffle: bool
        是否对数据shuffle，默认True
    entity_start_token: str
        用于在path中标记实体开头；如果设置为None，则不加任何token；默认为None
    entity_end_token: str
        用于在path中标记实体结尾；如果设置为None，则不加任何token；默认为None
    head_entity_start_token: str
        用于在path中标记头实体开头；如果设置为None，则不加任何token；默认为None；该选项不能和上面两条同时设置
    head_entity_end_token: str
        用于在path中标记头实体结尾；如果设置为None，则不加任何token；默认为None；该选项不能和上面两条同时设置
    tail_entity_start_token: str
        用于在path中标记尾实体开头；如果设置为None，则不加任何token；默认为None；该选项不能和上面两条同时设置
    tail_entity_end_token: str
        用于在path中标记尾实体结尾；如果设置为None，则不加任何token；默认为None；该选项不能和上面两条同时设置
    unknown_entity_token: str
        用于标记实体名缺失的实体（需要和处理的数据保持一致），默认为[unknown entity]
    separate_token: str
        用于在拼接的路径中用作分隔符，默认为[SEP]
    annotate_self_token: str
        当候选实体为中心实体时，用该token将其标记出来；如果设置为None，则表示不对self加任何token；默认为None
    """
    def __init__(self, file_path: str, split: str, bert_model_name_or_path: str, shuffle: bool = True, entity_start_token: str = None, entity_end_token: str = None, head_entity_start_token: str = None, head_entity_end_token: str = None, tail_entity_start_token: str = None, tail_entity_end_token: str = None, unknown_entity_token: str = "[unknown entity]", separate_token: str = "[SEP]", annotate_self_token: str = None, use_prompt: bool = False, mix_webred_data: bool = False, mix_webred_matching_data: bool = False, mix_bertrl_data: bool = False, use_logging=True):
        samples = json.load(open(file_path, "r"))
        if use_logging:
            logging.info(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
        else:
            print(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
        samples_for_matching = []
        for q_id, sample in samples.items():
            if sample["is_subgraph_empty"]:
                continue
            question = sample["question"]
            for candidate in sample["candidiates"]:  # a typo when processing data file
                question_with_prompt = question
                if len(candidate["converted_paths"]) > 0:
                    candidate_name = candidate["converted_paths"][0][-1]
                    if candidate_name != unknown_entity_token:
                        question_with_prompt = f"{question} ? is the correct answer {candidate_name} ?"
                samples_for_matching.append({
                    "question": question_with_prompt if use_prompt else question,
                    "paths": self._convert_and_combine_paths(candidate["converted_paths"], candidate["is_self"], annotate_self_token, entity_start_token, entity_end_token, head_entity_start_token, head_entity_end_token, tail_entity_start_token, tail_entity_end_token, separate_token),
                    "id": q_id,
                    "label": candidate["is_answer"],
                    "candi": candidate["candidate_id"]
                })
        if mix_webred_data and split == "train":
            assert shuffle, "Mixed WebRED data during training but not shuffle"
            webred_file = "./data/webred/webred_21_pretraining.json"
            webred_samples = json.load(open(webred_file, "r"))
            for sample in webred_samples:
                for item in sample["true"] + sample["false"]:
                    samples_for_matching.append({
                        "question": item["q"],
                        "paths": "; ".join(item["path"]),
                        "id": item["id"],
                        "label": item["label"],
                        "candi": item["candi"]
                    })
        if mix_webred_matching_data and split == "train":
            assert shuffle, "Mixed WebRED matching data during training but not shuffle"
            webred_file = "./data/webred_matching/webred_21_for_matching.json"
            webred_samples = json.load(open(webred_file, "r"))
            for sample in webred_samples:
                samples_for_matching.append({
                    "question": sample["text1"],
                    "paths": sample["text2"],
                    "id": "",
                    "label": sample["label"],
                    "candi": ""
                })
        if mix_bertrl_data and split == "train":
            assert shuffle, "Mixed BERTRL data during training but not shuffle"
            bertrl_file = "./data/freebase_pretraining_bertrl/webqsp/train.tsv"
            with open(bertrl_file, "r") as f:
                for idx, line in enumerate(f):
                    if idx >= 1000000:
                        break
                    label, _, _, text1, text2 = line.strip().split("\t")
                    samples_for_matching.append({
                        "question": text1,
                        "paths": text2,
                        "id": "",
                        "label": int(label),
                        "candi": ""
                    })
        if shuffle:
            random.shuffle(samples_for_matching)
        self.data = samples_for_matching
        self.num_data = len(self.data)
        self.special_tokens = [unknown_entity_token, separate_token]
        for special_token in (entity_start_token, entity_end_token, head_entity_start_token, head_entity_end_token, tail_entity_start_token, tail_entity_end_token, annotate_self_token):
            if special_token is not None:
                self.special_tokens.append(special_token)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name_or_path)
    
    def _convert_and_combine_paths(self, paths: List[List[str]], is_self: bool, annotate_self_token: str = None, entity_start_token: str = None, entity_end_token: str = None, head_entity_start_token: str = None, head_entity_end_token: str = None, tail_entity_start_token: str = None, tail_entity_end_token: str = None, separate_token: str = "[SEP]"):
        converted_paths = []
        if is_self and annotate_self_token is not None:
            converted_paths.append(annotate_self_token)
        for path in paths:
            tokens = []
            for idx, item in enumerate(path):
                if idx in (1, 3):  # relations
                    tokens.append(item.replace(".", " ").replace("_", " ").strip())
                else:  # entities
                    if entity_start_token is not None:
                        tokens.append(entity_start_token)
                    elif idx == 0 and head_entity_start_token is not None:
                        tokens.append(head_entity_start_token)
                    elif idx == len(path) - 1 and tail_entity_start_token is not None:
                        tokens.append(tail_entity_start_token)
                    
                    tokens.append(item.strip())
                    
                    if entity_end_token is not None:
                        tokens.append(entity_end_token)
                    elif idx == 0 and head_entity_end_token is not None:
                        tokens.append(head_entity_end_token)
                    elif idx == len(path) - 1 and tail_entity_end_token is not None:
                        tokens.append(tail_entity_end_token)
            converted_paths.append(" ".join(tokens))
        return f" {separate_token} ".join(converted_paths)
    
    def shuffle(self):
        random.shuffle(self.data)
        
    def get_batch(self, iteration, batch_size, max_length):
        batch_data = self.data[iteration * batch_size : (iteration + 1) * batch_size]
        
        q_ids = [sample["id"] for sample in batch_data]
        candidates = [sample["candi"] for sample in batch_data]
        labels = torch.tensor([sample["label"] for sample in batch_data]).long()
        
        bert_inputs = self.tokenizer.batch_encode_plus([[sample["question"], sample["paths"]] for sample in batch_data], padding=True, truncation=True, max_length=max_length, return_tensors="pt", return_token_type_ids=True)
        input_ids = bert_inputs["input_ids"]
        token_type_ids = bert_inputs["token_type_ids"]
        attention_mask = bert_inputs["attention_mask"]
        
        return input_ids, token_type_ids, attention_mask, labels, q_ids, candidates
    
    
class WebREDForPretrainingDataLoader(object):
    """
    对应WebRED处理之后的for pretraining的数据（data/webred/webred_xx_pretraining.json）。
    """
    def __init__(self, file_path: str, split: str, bert_model_name_or_path: str, shuffle: bool = True):
        samples = json.load(open(file_path, "r"))
        logging.info(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
        samples_for_matching = []
        for sample in samples:
            for item in sample["true"] + sample["false"]:
                samples_for_matching.append({
                    "question": item["q"],
                    "paths": "; ".join(item["path"]),
                    "id": item["id"],
                    "label": item["label"],
                    "candi": item["candi"]
                })
        if shuffle:
            random.shuffle(samples_for_matching)
        self.data = samples_for_matching
        self.num_data = len(self.data)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name_or_path)
        
    def shuffle(self):
        random.shuffle(self.data)
        
    def get_batch(self, iteration, batch_size, max_length):
        batch_data = self.data[iteration * batch_size : (iteration + 1) * batch_size]
        
        q_ids = [sample["id"] for sample in batch_data]
        candidates = [sample["candi"] for sample in batch_data]
        labels = torch.tensor([sample["label"] for sample in batch_data]).long()
        
        bert_inputs = self.tokenizer.batch_encode_plus([[sample["question"], sample["paths"]] for sample in batch_data], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = bert_inputs["input_ids"]
        token_type_ids = bert_inputs["token_type_ids"]
        attention_mask = bert_inputs["attention_mask"]
        
        return input_ids, token_type_ids, attention_mask, labels, q_ids, candidates
    

class OriginalWebQSPWithPromptDataLoader(OriginalWebQSPDataLoader):
    """
    对应瑛瑶处理的那版数据（full_noconstrain），加上了自然语言提示词。
    """
    def __init__(self, file_path: str, split: str, bert_model_name_or_path: str, shuffle: bool = True):
        samples = json.load(open(file_path, "r"))
        logging.info(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
        samples_for_matching = []
        for sample in samples:
            for item in sample["true"] + sample["false"]:
                if len(item["path"]) > 0:
                    candi = item["path"][0].strip().split("~")[-1]
                    question = item["q"] + f", is {candi} the correct answer ?"
                else:
                    question = item["q"]
                samples_for_matching.append({
                    "question": question,
                    "paths": "; ".join(item["path"]),
                    "id": item["id"],
                    "label": item["label"],
                    "candi": item["candi"]
                })
        if shuffle:
            random.shuffle(samples_for_matching)
        self.data = samples_for_matching
        self.num_data = len(self.data)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name_or_path)
        
        
class MixedWebQSPWebREDDataLoader(OriginalWebQSPDataLoader):
    """
    在训练集中，会把WebQSP和WebRED的数据混合到一起。
    """
    def __init__(self, file_path: str, split: str, bert_model_name_or_path: str, shuffle: bool = True):
        if split == "train":
            webqsp_samples = json.load(open(file_path, "r"))
            logging.info(f"Reading {split} split from {file_path}, number of samples: {len(webqsp_samples)}")
            webred_file = "./data/webred/webred_21_pretraining.json"
            webred_samples = json.load(open(webred_file, "r"))
            logging.info(f"Reading {split} split from {webred_file}, number of samples: {len(webred_samples)}")
            samples = webqsp_samples + webred_samples
            logging.info(f"Merging WebQSP and WebRED data in train split, total number of samples: {len(samples)}")
        else:
            samples = json.load(open(file_path, "r"))
            logging.info(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
            
        samples_for_matching = []
        for sample in samples:
            for item in sample["true"] + sample["false"]:
                samples_for_matching.append({
                    "question": item["q"],
                    "paths": "; ".join(item["path"]),
                    "id": item["id"],
                    "label": item["label"],
                    "candi": item["candi"]
                })
        if shuffle:
            random.shuffle(samples_for_matching)
        self.data = samples_for_matching
        self.num_data = len(self.data)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name_or_path)


class SplitedPathWebQSPDataLoader(object):
    """
    对应新处理的那版数据，把大部分缺失的实体名补全了（但仍有少数找不到实体名的，表示为[unknown entity]），路径：./data/webqsp_name_expanded
    这一版DataLoader将把路径切分，分别和question匹配后送入BERT。
    
    Args
    ----
    file_path: str
        读取的数据文件路径；
    split: str
        对应属于数据集的哪个split
    bert_model_name_or_path: str
        预训练语言模型的路径，用于加载tokenizer
    max_batch_size: int
        训练时最大的batch size，会在训练时用来控制一个batch使用多少实例，保证其所有path的拼接不超过这个值
    shuffle: bool
        是否对数据shuffle，默认True
    entity_start_token: str
        用于在path中标记实体开头；如果设置为None，则不加任何token；默认为None
    entity_end_token: str
        用于在path中标记实体结尾；如果设置为None，则不加任何token；默认为None
    head_entity_start_token: str
        用于在path中标记头实体开头；如果设置为None，则不加任何token；默认为None；该选项不能和上面两条同时设置
    head_entity_end_token: str
        用于在path中标记头实体结尾；如果设置为None，则不加任何token；默认为None；该选项不能和上面两条同时设置
    tail_entity_start_token: str
        用于在path中标记尾实体开头；如果设置为None，则不加任何token；默认为None；该选项不能和上面两条同时设置
    tail_entity_end_token: str
        用于在path中标记尾实体结尾；如果设置为None，则不加任何token；默认为None；该选项不能和上面两条同时设置
    unknown_entity_token: str
        用于标记实体名缺失的实体（需要和处理的数据保持一致），默认为[unknown entity]
    use_aggregate: bool
        在训练时使用聚合计算损失or直接作为二分类损失，默认直接作为二分类损失
    """
    def __init__(self, file_path: str, split: str, bert_model_name_or_path: str, max_batch_size: int, shuffle: bool = True, entity_start_token: str = None, entity_end_token: str = None, head_entity_start_token: str = None, head_entity_end_token: str = None, tail_entity_start_token: str = None, tail_entity_end_token: str = None, unknown_entity_token: str = "[unknown entity]", use_aggregate: bool = False):
        samples = json.load(open(file_path, "r"))
        logging.info(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
        samples_for_matching = []
        if use_aggregate:
            for q_id, sample in samples.items():
                if sample["is_subgraph_empty"]:
                    continue
                question = sample["question"]
                for candidate in sample["candidiates"]:  # a typo when processing data file
                    samples_for_matching.append({
                        "question": question,
                        "paths": self._convert_paths(candidate["converted_paths"], entity_start_token, entity_end_token, head_entity_start_token, head_entity_end_token, tail_entity_start_token, tail_entity_end_token),
                        "id": q_id,
                        "label": candidate["is_answer"],
                        "candi": candidate["candidate_id"]
                    })
        else:
            for q_id, sample in samples.items():
                if sample["is_subgraph_empty"]:
                    continue
                question = sample["question"]
                for candidate in sample["candidiates"]:  # a typo when processing data file
                    for path in self._convert_paths(candidate["converted_paths"], entity_start_token, entity_end_token, head_entity_start_token, head_entity_end_token, tail_entity_start_token, tail_entity_end_token):
                        samples_for_matching.append({
                            "question": question,
                            "paths": path,
                            "id": q_id,
                            "label": candidate["is_answer"],
                            "candi": candidate["candidate_id"]
                        })
        if shuffle:
            random.shuffle(samples_for_matching)
        self.data = samples_for_matching
        self.special_tokens = [unknown_entity_token]
        for special_token in (entity_start_token, entity_end_token, head_entity_start_token, head_entity_end_token, tail_entity_start_token, tail_entity_end_token):
            if special_token is not None:
                self.special_tokens.append(special_token)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name_or_path)
        self.max_batch_size = max_batch_size
        self.use_aggregate = use_aggregate
        if use_aggregate:
            for sample in self.data:
                assert len(sample["paths"]) <= max_batch_size  # ensure no size of paths exceeds the max_batch_size (or it can cause the infinite loop)
            self._generate_steps()
            self.num_data = (len(self.steps) - 1) * max_batch_size
        else:
            self.num_data = len(self.data)
        logging.info(f"Number of training samples after processing: {self.num_data}")
        
    def _generate_steps(self):
        self.steps = []
        self.offsets = []
        current_step = 0
        while current_step < len(self.data):
            self.steps.append(current_step)
            accumulated_num_samples = 0
            offsets = []
            while current_step < len(self.data) and accumulated_num_samples + len(self.data[current_step]["paths"]) <= self.max_batch_size:
                offsets.append(accumulated_num_samples)
                accumulated_num_samples += len(self.data[current_step]["paths"])
                current_step += 1
            offsets.append(accumulated_num_samples)  # added the (n+1)-th item for list slice in the training loop
            self.offsets.append(offsets)
        self.steps.append(len(self.data))  # added the (n+1)-th item for list slice in get_batch method
    
    def _convert_paths(self, paths: List[List[str]], entity_start_token: str = None, entity_end_token: str = None, head_entity_start_token: str = None, head_entity_end_token: str = None, tail_entity_start_token: str = None, tail_entity_end_token: str = None):
        converted_paths = []
        for path in paths:
            tokens = []
            for idx, item in enumerate(path):
                if idx in (1, 3):  # relations
                    tokens.append(item.replace(".", " ").replace("_", " ").strip())
                else:  # entities
                    if entity_start_token is not None:
                        tokens.append(entity_start_token)
                    elif idx == 0 and head_entity_start_token is not None:
                        tokens.append(head_entity_start_token)
                    elif idx == len(path) - 1 and tail_entity_start_token is not None:
                        tokens.append(tail_entity_start_token)
                    
                    tokens.append(item.strip())
                    
                    if entity_end_token is not None:
                        tokens.append(entity_end_token)
                    elif idx == 0 and head_entity_end_token is not None:
                        tokens.append(head_entity_end_token)
                    elif idx == len(path) - 1 and tail_entity_end_token is not None:
                        tokens.append(tail_entity_end_token)
            converted_paths.append(" ".join(tokens))
        return converted_paths
    
    def shuffle(self):
        random.shuffle(self.data)
        
    def get_batch(self, iteration, batch_size, max_length):
        if self.use_aggregate:
            batch_data = self.data[self.steps[iteration]: self.steps[iteration + 1]]
            offsets = self.offsets[iteration]  # path offsets for candidates in this batch
        else:
            batch_data = self.data[iteration * batch_size : (iteration + 1) * batch_size]
        
        q_ids = [sample["id"] for sample in batch_data]
        candidates = [sample["candi"] for sample in batch_data]
        labels = torch.tensor([sample["label"] for sample in batch_data]).long()
        
        if self.use_aggregate:
            texts_to_encode = []
            for sample in batch_data:
                question = sample["question"]
                for path in sample["paths"]:
                    texts_to_encode.append([question, path])
            bert_inputs = self.tokenizer.batch_encode_plus(texts_to_encode, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        else:
            bert_inputs = self.tokenizer.batch_encode_plus([[sample["question"], sample["paths"]] for sample in batch_data], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = bert_inputs["input_ids"]
        token_type_ids = bert_inputs["token_type_ids"]
        attention_mask = bert_inputs["attention_mask"]
        
        if self.use_aggregate:
            assert len(offsets) - 1 == len(labels) == len(q_ids) == len(candidates)
            return input_ids, token_type_ids, attention_mask, offsets, labels, q_ids, candidates
        else:
            return input_ids, token_type_ids, attention_mask, labels, q_ids, candidates
        

class WebREDMatchingDataLoader(object):
    """
    两句话判断是否具有同一个关系的预训练任务，路径：./data/webred_matching
    """
    def __init__(self, file_path: str, split: str, bert_model_name_or_path: str, shuffle: bool = True):
        samples = json.load(open(file_path, "r"))
        logging.info(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
        if shuffle:
            random.shuffle(samples)
        self.data = samples
        self.num_data = len(self.data)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name_or_path)
        
    def shuffle(self):
        random.shuffle(self.data)
        
    def get_batch(self, iteration, batch_size, max_length):
        batch_data = self.data[iteration * batch_size : (iteration + 1) * batch_size]
        
        labels = torch.tensor([sample["label"] for sample in batch_data]).long()
        bert_inputs = self.tokenizer.batch_encode_plus([[sample["text1"], sample["text2"]] for sample in batch_data], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = bert_inputs["input_ids"]
        token_type_ids = bert_inputs["token_type_ids"]
        attention_mask = bert_inputs["attention_mask"]
        
        return input_ids, token_type_ids, attention_mask, labels
    

class WebREDMaskedTokenPredictionDataLoader(object):
    """
    基于WebRED做MLM的任务。每个样本中，除子图和实体、高频词（top 3%）的词以外，剩余的wordpiece中，25%将被mask。
    """
    def __init__(self, file_path: str, split: str, bert_model_name_or_path: str, shuffle: bool = True):
        samples = json.load(open(file_path, "r"))
        
        filtered_samples = []
        for sample in samples:
            if sample["num_pos_raters"] > sample["num_raters"] - sample["num_pos_raters"]:
                filtered_samples.append(sample)
        samples = filtered_samples
        
        logging.info(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
        
        # 统计高频词
        self.tokenizer = RobertaTokenizerFast.from_pretrained(bert_model_name_or_path)
        self.compute_high_frequency_tokens(samples)
        
        if shuffle:
            random.shuffle(samples)
        self.data = samples
        self.num_data = len(self.data)
        
    def compute_high_frequency_tokens(self, samples):
        counter = defaultdict(int)
        for sample in samples:
            text = sample["annotated_sentence_text"]
            tokens = self.tokenizer.tokenize(text)
            for token in tokens:
                counter[token] += 1
        token_and_frequency = [(token, count) for token, count in counter.items()]
        token_and_frequency = sorted(token_and_frequency, key=lambda x: x[1], reverse=True)
        total_num_tokens = len(token_and_frequency)
        num_high_frequency_tokens = min(int(total_num_tokens * 0.02), 100)
        high_frequency_tokens = [token for token, _ in token_and_frequency[:num_high_frequency_tokens]]
        self.high_freq_token_ids = set(self.tokenizer.convert_tokens_to_ids(high_frequency_tokens))
        
    def shuffle(self):
        random.shuffle(self.data)
        
    def get_batch(self, iteration, batch_size, max_length):
        batch_data = self.data[iteration * batch_size : (iteration + 1) * batch_size]
        
        def gen_invalid_spans(text, pattern):
            spans = []
            start = 0
            while start < len(text):
                left_idx = text.find(pattern[0], start)
                if left_idx == -1:
                    break
                start = left_idx + len(pattern[0])
                right_idx = text.find(pattern[1], start)
                if right_idx == -1:
                    break
                start = right_idx + len(pattern[1])
                spans.append((left_idx, right_idx + len(pattern[1])))
            return spans
        
        batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = [], [], [], []
        for sample in batch_data:
            text = sample["annotated_sentence_text"]
            invalid_spans = gen_invalid_spans(text, ["SUBJ{", "}"]) + gen_invalid_spans(text, ["OBJ{", "}"])
            relation = sample["relation_name"]
            source = sample["source_name"]
            target = sample["target_name"]
            encoded = self.tokenizer.encode_plus([text, f"{source}~{relation}~{target}"], return_offsets_mapping=True, padding="max_length", truncation=True, max_length=max_length)
            masked_idxes = []
            for idx, (input_id, offset) in enumerate(zip(encoded["input_ids"], encoded["offset_mapping"])):
                if input_id == 102:
                    break
                if offset[1] == 0:  # for special tokens, like [CLS]
                    continue
                if input_id in self.high_freq_token_ids:  # for high frequency tokens
                    continue
                if any(offset[0] >= start and offset[1] <= end for start, end in invalid_spans):  # for entities like SUBJ{xxx} and OBJ{xxx}
                    continue
                masked_idxes.append(idx)
            random.shuffle(masked_idxes)
            num_masked_tokens = math.ceil(len(masked_idxes) * 0.25)
            masked_idxes = masked_idxes[:num_masked_tokens]
            labels = len(encoded["input_ids"]) * [-100]  # -100 indicates the ignored index when calculate cross-entropy los13
            for idx in masked_idxes:
                labels[idx] = encoded["input_ids"][idx]
                encoded["input_ids"][idx] = self.tokenizer.mask_token_id
            batch_input_ids.append(encoded["input_ids"])
            batch_token_type_ids.append(encoded["token_type_ids"])
            batch_attention_mask.append(encoded["attention_mask"])
            batch_labels.append(labels)
        batch_input_ids = torch.tensor(batch_input_ids)
        batch_token_type_ids = torch.tensor(batch_token_type_ids)
        batch_attention_mask = torch.tensor(batch_attention_mask)
        batch_labels = torch.tensor(batch_labels)
                
        return batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_labels
    

class BERTRLMatchingDataLoader(object):
    """
    用于BERTRL的预训练（也是一个简单的匹配任务）。处理好的数据路径：./data/freebase_pretraining_bertrl
    """
    def __init__(self, file_path: str, split: str, bert_model_name_or_path: str, shuffle: bool = True):
        samples = open(file_path).readlines()
        logging.info(f"Reading {split} split from {file_path}, number of samples: {len(samples)}")
        if shuffle:
            random.shuffle(samples)
        self.data = samples
        self.num_data = len(self.data)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_model_name_or_path)
        
    def shuffle(self):
        random.shuffle(self.data)
        
    def get_batch(self, iteration, batch_size, max_length):
        batch_data = self.data[iteration * batch_size : (iteration + 1) * batch_size]
        
        labels = []
        texts_to_encode = []
        for line in batch_data:
            label, _, _, text1, text2 = line.strip().split("\t")
            labels.append(int(label))
            texts_to_encode.append([text1, text2])
        
        labels = torch.tensor(labels).long()
        bert_inputs = self.tokenizer.batch_encode_plus(texts_to_encode, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = bert_inputs["input_ids"]
        token_type_ids = bert_inputs["token_type_ids"]
        attention_mask = bert_inputs["attention_mask"]
        
        return input_ids, token_type_ids, attention_mask, labels