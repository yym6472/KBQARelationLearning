import os
import sys
import math
import json
import tqdm
import torch
import random
import argparse
import my_logging as logging
import shutil
import numpy as np

from tensorboardX import SummaryWriter
from data_loader import OriginalWebQSPDataLoader, NameExpandedWebQSPDataLoader, WebREDForPretrainingDataLoader, OriginalWebQSPWithPromptDataLoader, MixedWebQSPWebREDDataLoader, SplitedPathWebQSPDataLoader
from model import BertForKBQA


def parse_args():
    """
    Argument settings.
    """
    # arguments for experiments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducing experimental results")
    parser.add_argument("--task", type=str, default="kbqa", choices=["kbqa", "kbqa_name_expanded", "webred", "kbqa_with_prompt", "mixed_kbqa", "kbqa_multi_paths"], help="Define the type of training task (or dataset)")
    parser.add_argument("--dataset", type=str, default="ori_webqsp", choices=["ori_webqsp", "webqsp", "cwq"], help="The name (id) of dataset")
    parser.add_argument("--model_name_or_path", type=str, default="./bert-base-uncased/", help="The model path or model name of pre-trained model")
    parser.add_argument("--continue_training_path", type=str, default=None, help="The checkpoint path to load for continue training")
    parser.add_argument("--model_path", type=str, required=True, help="Custom output dir")
    parser.add_argument("--output_filename", type=str, default="evaluate_test_origin_fixed.json")
    parser.add_argument("--tofix_filename", type=str, default="evaluate_test_origin.json")
    parser.add_argument("--force_del", action="store_true", help="Delete the existing save_path and do not report an error")
    parser.add_argument("--tensorboard_log_dir", type=str, default=None, help="Custom tensorboard log dir")
    parser.add_argument("--data_path", type=str, default="./data/webqsp_yingyao", help="The data dir")
    parser.add_argument("--train_filename", type=str, default="train.json", help="The filename of train dataset")
    parser.add_argument("--dev_filename", type=str, default="dev.json", help="The filename of dev dataset")
    parser.add_argument("--test_filename", type=str, default="test.json", help="The filename of test dataset")
    
    # arguments for amp
    parser.add_argument("--use_apex_amp", action="store_true", help="Use apex amp or not")
    parser.add_argument("--apex_amp_opt_level", type=str, default=None, help="The opt_level argument in apex amp")
    
    # arguments for models
    parser.add_argument("--dropout", type=float, default=0.2, help="The dropout rate of BERT derived representations")
    parser.add_argument("--threshold", type=float, default=None, help="The threshold for binary classification. If set to None, it will be automatically inferred from the dev set")
    parser.add_argument("--entity_start_token", type=str, default=None, help="The token to annotate the start of an entity")
    parser.add_argument("--entity_end_token", type=str, default=None, help="The token to annotate the end of an entity")
    parser.add_argument("--head_entity_start_token", type=str, default=None, help="The token to annotate the start of a head entity")
    parser.add_argument("--head_entity_end_token", type=str, default=None, help="The token to annotate the end of a head entity")
    parser.add_argument("--tail_entity_start_token", type=str, default=None, help="The token to annotate the start of a tail entity")
    parser.add_argument("--tail_entity_end_token", type=str, default=None, help="The token to annotate the end of a tail entity")
    parser.add_argument("--unknown_entity_token", type=str, default="[unknown entity]", help="The token to annotate entities with no entity name found")
    parser.add_argument("--separate_token", type=str, default="[path separator]", help="The token to separator multiple paths in the combined paths")
    parser.add_argument("--annotate_self_token", type=str, default=None, help="The token to annotate that the candidate entity is equal to the topic entity")
    parser.add_argument("--use_aggregate", action="store_true", help="Use path aggregate when training")
    parser.add_argument("--use_prompt", action="store_true", help="Use prompt for question when training KBQA")
    parser.add_argument("--mix_webred_data", action="store_true", help="Mix WebRED data when training KBQA")
    parser.add_argument("--mix_webred_matching_data", action="store_true", help="Mix WebRED matching data when training KBQA")

    # arguments for training (or data preprocessing)
    parser.add_argument("--batch_size", type=int, default=128, help="Training mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="The learning rate")
    parser.add_argument("--evaluation_steps", type=int, default=1000, help="The steps between every evaluations")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The max sequence length")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="The max norm of gradient to apply gradient clip")
    parser.add_argument("--patience", type=int, default=None, help="Patience for early stop")
    parser.add_argument("--metric", type=str, default="hits", choices=["hits", "f1", "accuracy"], help="The metric to select the best model")
    
    return parser.parse_args()

def check_args(args):
    if args.task in ("kbqa", "kbqa_with_prompt", "mixed_kbqa", "kbqa_name_expanded"):
        assert args.metric in ("hits", "f1"), "KBQA task only support hits and f1 as metric"
    elif args.task in ("webred",):
        assert args.metric in ("accuracy",), "WebRED task only support accuracy as metric"

def set_seed(seed: int, for_multi_gpu: bool):
    """
    Added script to set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if for_multi_gpu:
        torch.cuda.manual_seed_all(seed)

def prepare_for_training(args):
    """
    - Record the training command and arguments
    """
    logging_stream = sys.stdout
    logging.basicConfig(logging_stream)
    
    logging.info(f"Training arguments:\n{json.dumps(args.__dict__, indent=4, ensure_ascii=False)}")

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

def evaluate_kbqa(args, data_loader, model, batch_size, max_seq_length, thresholds=None, log_file=None, multi_paths=False):
    """
    Calculate f1 and hits@1 metrics for dev and test dataset.
    
    Args
    ----
    thresholds: List[float] or float or None
        All thresholds to try, the best f1 results will be returned. If set to None, [0.001, 0.002, ..., 0.500]
        will be set as the default value.
    """
    # loading answer entities
    if args.dataset == "cwq":
        answers_mapping = json.load(open("./data/NSM_data/CWQ/answers_mapping.json"))
    elif args.dataset == "webqsp":
        answers_mapping = json.load(open("./data/NSM_data/webqsp/answers_mapping.json"))
    elif args.dataset == "ori_webqsp":
        webqsp_processed = json.load(open("./data/GraftNet/preprocessing/scratch/webqsp_processed.json", "r"))
        answers_mapping = {item["QuestionId"]: [each["freebaseId"] for each in item["Answers"]] for item in webqsp_processed}
    
    question_wise_scores = {k: v["scores"] for k, v in json.load(open(os.path.join(args.model_path, args.tofix_filename))).items()}
    
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
            "loss": 0.0
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

def main(args):
    check_args(args)
    prepare_for_training(args)
    
    # evaluate on test dataset
    logging.info("Evaluation on test dataset...")
    if args.metric in ("hits", "f1"):
        threshold, p, r, f1, hits, loss = evaluate_kbqa(args, None, None, args.batch_size, args.max_seq_length, thresholds=0.2, log_file=os.path.join(args.model_path, args.output_filename), multi_paths=(args.task == "kbqa_multi_paths" and args.use_aggregate))
        logging.info(f"Results: f1 = {f1:.6f} (threshold = {threshold:.6f}), hits@1 = {hits:.6f}")
    elif args.metric == "accuracy":
        acc = evaluate_accuracy(None, None, args.batch_size, args.max_seq_length, log_file=os.path.join(args.model_path, args.output_filename))
        logging.info(f"Results: acc = {acc:.6f}")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)