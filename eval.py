"""
python3 eval.py --model_path ./output/bert_newdata_headtail_webred_mix_seed2 --test_file ./data/webqsp_name_expanded/test_2000.json --add_head_tail_mark --log_filename evaluate_test_2000.json --threshold 0.2
"""


import os
import torch
import tqdm
import math
import json
import argparse

from model import BertForKBQA
from data_loader import NameExpandedWebQSPDataLoader


def cal_accuracy(preds, labels):
    """
    Calculate the accuracy for binary classification (only used during the training phase).
    """
    return (preds == labels).long().sum().item() / len(preds)

def get_one_f1(entities, dist, eps, answers):
    """
    Copy from GraftNet.
    """
    correct = 0.0
    total = 0.0
    best_entity = -1
    max_prob = 0.0
    preds = []
    for entity in entities:
        if dist[entity] > max_prob or math.fabs(dist[entity] - max_prob) < 1e-9:  # 当分数相同时，更新为靠后的那个，以解决过拟合时始终hit的问题
            max_prob = dist[entity]
            best_entity = entity
        if dist[entity] > eps:
            preds.append(entity)
    
    return cal_eval_metric(best_entity, preds, answers)

def cal_eval_metric(best_pred, preds, answers):
    """
    Copy from GraftNet.
    """
    correct, total = 0.0, 0.0
    for entity in preds:
        if entity in answers:
            correct += 1
        total += 1
    if len(answers) == 0:
        if total == 0:
            return 1.0, 1.0, 1.0, 1.0 # precision, recall, f1, hits
        else:
            return 0.0, 1.0, 0.0, 1.0 # precision, recall, f1, hits
    else:
        hits = float(best_pred in answers)
        if total == 0:
            return 1.0, 0.0, 0.0, hits # precision, recall, f1, hits
        else:
            precision, recall = correct / total, correct / len(answers)
            f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision != 0 and recall != 0 else 0.0
            return precision, recall, f1, hits

def evaluate_kbqa(data_loader, model, batch_size, max_seq_length, thresholds=None, log_file=None, multi_paths=False):
    """
    Calculate f1 and hits@1 metrics for dev and test dataset.
    
    Args
    ----
    thresholds: List[float] or float or None
        All thresholds to try, the best f1 results will be returned. If set to None, [0.001, 0.002, ..., 0.500]
        will be set as the default value.
    """
    # loading answer entities
    webqsp_processed = json.load(open("./data/GraftNet/preprocessing/scratch/webqsp_processed.json", "r"))
    answers_mapping = {item["QuestionId"]: [each["freebaseId"] for each in item["Answers"]] for item in webqsp_processed}
    
    # evaluate by iterating data loader
    all_scores, all_q_ids, all_labels, all_candidates, losses = [], [], [], [], []
    model.eval()
    for batch_idx in tqdm.tqdm(range(math.ceil(data_loader.num_data / batch_size))):
        batch = data_loader.get_batch(batch_idx, batch_size, max_seq_length)
        batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]
        if multi_paths:
            input_ids, token_type_ids, attention_mask, offsets, labels, q_ids, candidates = batch
            batch_size = input_ids.shape[0]
            def expand_by_offsets(max_size, offsets, data_list):
                results = []
                current_idx = -1
                for idx in range(max_size):
                    if idx >= offsets[current_idx + 1]:
                        current_idx += 1
                    results.append(data_list[current_idx])
                return results
            labels = torch.tensor(expand_by_offsets(input_ids.shape[0], offsets, labels.tolist())).long().cuda()
            q_ids = expand_by_offsets(input_ids.shape[0], offsets, q_ids)
            candidates = expand_by_offsets(input_ids.shape[0], offsets, candidates) 
        else:
            input_ids, token_type_ids, attention_mask, labels, q_ids, candidates = batch
        
        logits = model(input_ids, token_type_ids, attention_mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        all_scores.extend(torch.sigmoid(logits).tolist())
        all_q_ids.extend(q_ids)
        all_labels.extend(labels.tolist())
        all_candidates.extend(candidates)
        losses.append(loss.item())
       
    question_wise_scores = {}
    all_questions = [item["question"] for item in data_loader.data]
    all_paths = [item["paths"] for item in data_loader.data]
    for score, q_id, label, candidate, question, paths in zip(all_scores, all_q_ids, all_labels, all_candidates, all_questions, all_paths):
        if q_id not in question_wise_scores:
            question_wise_scores[q_id] = []
        question_wise_scores[q_id].append((candidate, score, label, question, paths))
    
    # calculate the hits@1, f1, precision and recall metrics
    if isinstance(thresholds, float):  # only single (specified) threshold to run
        thresholds = [thresholds]
    elif thresholds is None:  # default value: try from 0.001 to 0.500
        thresholds = [float((i + 1) / 1000) for i in range(500)]
    
    best_overall, best_evaluate_details = None, None
    for threshold in thresholds:
        all_hits, all_precision, all_recall, all_f1 = [], [], [], []
        evaluate_details = {}
        for q_id, scores in question_wise_scores.items():
            entities = set()
            dist = {}
            answers = answers_mapping[q_id]
            for candidate, score, label, _, _ in scores:
                entities.add(candidate)
                if (candidate not in dist) or (candidate in dist and score > dist[candidate]):
                    dist[candidate] = score
            p, r, f1, hits = get_one_f1(entities, dist, threshold, answers)
            all_hits.append(hits)
            all_precision.append(p)
            all_recall.append(r)
            all_f1.append(f1)
            evaluate_details[q_id] = {
                "precision": p,
                "recall": r,
                "f1": f1,
                "hits@1": hits,
                "scores": sorted(scores, key=lambda x: (x[2], x[1]), reverse=True)
            }
        overall = {
            "threshold": threshold,
            "precision": sum(all_precision) / len(all_precision),
            "recall": sum(all_recall) / len(all_recall),
            "f1": sum(all_f1) / len(all_f1),
            "hits@1": sum(all_hits) / len(all_hits),
            "loss": sum(losses) / len(losses)
        }
        if best_overall is None or overall["f1"] > best_overall["f1"]:
            best_overall = overall
            best_evaluate_details = evaluate_details
    
    # log and return
    if log_file is not None:
        json.dump(best_overall, open(log_file[:-5] + ".overall.json", "w"), indent=4)
        json.dump(best_evaluate_details, open(log_file, "w"), indent=4)
        
    return (best_overall[key] for key in ("threshold", "precision", "recall", "f1", "hits@1", "loss"))

def evaluate_accuracy(data_loader, model, batch_size, max_seq_length, log_file=None):
    """
    Calculate accuracy for evaluation.
    """
    # record the `q_id -> question & path` mapping
    question_and_path_dict = {}
    for item in data_loader.data:
        question = item["question"]
        paths = item["paths"]
        q_id = item["id"]
        assert q_id not in question_and_path_dict
        question_and_path_dict[q_id] = {
            "question": question,
            "paths": paths
        }
        
    # evaluate by iterating data loader
    all_scores, all_q_ids, all_labels, losses = [], [], [], []
    model.eval()
    for batch_idx in tqdm.tqdm(range(math.ceil(data_loader.num_data / batch_size))):
        batch = data_loader.get_batch(batch_idx, batch_size, max_seq_length)
        batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]
        input_ids, token_type_ids, attention_mask, labels, q_ids, _ = batch
        
        logits = model(input_ids, token_type_ids, attention_mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        all_scores.extend(torch.sigmoid(logits).tolist())
        all_q_ids.extend(q_ids)
        all_labels.extend(labels.tolist())
        losses.append(loss.item())
    
    # calculate accuracy
    question_wise_scores = {}
    for score, q_id, label in zip(all_scores, all_q_ids, all_labels):
        assert q_id not in question_wise_scores
        question_wise_scores[q_id] = {
            "score": score,
            "label": label,
            "accuracy": int(int(score > 0.5) == label),
            "text": question_and_path_dict[q_id]["question"],
            "path": question_and_path_dict[q_id]["paths"]
        }
    
    all_acc = [item["accuracy"] for item in question_wise_scores.values()]
    overall = {
        "accuracy": sum(all_acc) / len(all_acc)
    }
    
    # log and return
    if log_file is not None:
        json.dump(overall, open(log_file[:-5] + ".overall.json", "w"), indent=4)
        json.dump(question_wise_scores, open(log_file, "w"), indent=4)
        
    return overall["accuracy"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="The output path of the model to evaluate")
    parser.add_argument("--model_name_or_path", type=str, default="./bert-base-uncased/", help="The BERT model name or path")
    parser.add_argument("--test_file", type=str, required=True, help="The file of the test dataset")
    parser.add_argument("--add_entity_mark", action="store_true")
    parser.add_argument("--add_head_tail_mark", action="store_true")
    parser.add_argument("--entity_start_token", type=str, default=None, help="The token to annotate the start of an entity")
    parser.add_argument("--entity_end_token", type=str, default=None, help="The token to annotate the end of an entity")
    parser.add_argument("--head_entity_start_token", type=str, default=None, help="The token to annotate the start of a head entity")
    parser.add_argument("--head_entity_end_token", type=str, default=None, help="The token to annotate the end of a head entity")
    parser.add_argument("--tail_entity_start_token", type=str, default=None, help="The token to annotate the start of a tail entity")
    parser.add_argument("--tail_entity_end_token", type=str, default=None, help="The token to annotate the end of a tail entity")
    parser.add_argument("--unknown_entity_token", type=str, default="[unknown entity]", help="The token to annotate entities with no entity name found")
    parser.add_argument("--separate_token", type=str, default="[path separator]", help="The token to separator multiple paths in the combined paths")
    parser.add_argument("--annotate_self_token", type=str, default=None, help="The token to annotate that the candidate entity is equal to the topic entity")
    parser.add_argument("--use_prompt", action="store_true", help="Use prompt for question when training KBQA")
    parser.add_argument("--batch_size", type=int, default=128, help="Training mini-batch size")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The max sequence length")
    parser.add_argument("--log_filename", type=str, default="evaluate_test.json")
    parser.add_argument("--threshold", type=float, default=None)
    
    args = parser.parse_args()
    
    if args.add_entity_mark:
        args.entity_start_token = "[start of entity]"
        args.entity_end_token = "[end of entity]"
    if args.add_head_tail_mark:
        args.head_entity_start_token = "[start of head entity]"
        args.head_entity_end_token = "[end of head entity]"
        args.tail_entity_start_token = "[start of tail entity]"
        args.tail_entity_end_token = "[end of tail entity]"
    return args
    

def main(args):
    test_loader = NameExpandedWebQSPDataLoader(args.test_file, "test", args.model_name_or_path, shuffle=False, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, separate_token=args.separate_token, annotate_self_token=args.annotate_self_token, use_prompt=args.use_prompt, mix_webred_data=False, use_logging=False)
    model = BertForKBQA(args.model_name_or_path, dropout=0.2).cuda()
    model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin")))
    model.cuda()
    threshold, p, r, f1, hits, loss = evaluate_kbqa(test_loader, model, args.batch_size, args.max_seq_length, thresholds=args.threshold, log_file=os.path.join(args.model_path, args.log_filename), multi_paths=False)
    print(f"Results: f1 = {f1:.6f} (threshold = {threshold:.6f}), hits@1 = {hits:.6f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)