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
from data_loader import OriginalWebQSPDataLoader, NameExpandedWebQSPDataLoader, WebREDForPretrainingDataLoader, OriginalWebQSPWithPromptDataLoader, MixedWebQSPWebREDDataLoader, SplitedPathWebQSPDataLoader, PWNameExpandedWebQSPDataLoader
from model import BertForKBQA


def parse_args():
    """
    Argument settings.
    """
    # arguments for experiments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducing experimental results")
    parser.add_argument("--task", type=str, default="kbqa", choices=["kbqa", "kbqa_name_expanded", "webred", "kbqa_with_prompt", "mixed_kbqa", "kbqa_multi_paths", "kbqa_name_expanded_pairwise"], help="Define the type of training task (or dataset)")
    parser.add_argument("--margin", type=float, required=False, default=0.3, help="Pairwise loss margin")
    parser.add_argument("--dataset", type=str, default="ori_webqsp", choices=["ori_webqsp", "webqsp", "cwq"], help="The name (id) of dataset")
    parser.add_argument("--model_name_or_path", type=str, default="./bert-base-uncased/", help="The model path or model name of pre-trained model")
    parser.add_argument("--continue_training_path", type=str, default=None, help="The checkpoint path to load for continue training")
    parser.add_argument("--model_save_path", type=str, required=True, help="Custom output dir")
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
    parser.add_argument("--mix_webred_data", type=str, default=None, help="Mix WebRED data when training KBQA")
    parser.add_argument("--mix_webred_matching_data", type=str, default=None, help="Mix WebRED matching data when training KBQA")
    parser.add_argument("--mix_bertrl_data", type=str, default=None, help="Mix BERTRL data when training KBQA")
    parser.add_argument("--bertrl_max_samples", type=int, default=1000000, help="The maximum BERTRL data")

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
    - Initialize the random seed
    - Create the output directory
    - Record the training command and arguments
    """
    set_seed(args.seed, for_multi_gpu=False)
    if os.path.exists(args.model_save_path):
        if args.force_del:
            shutil.rmtree(args.model_save_path)
        else:
            raise ValueError("Existing output directory for saving model")
    os.mkdir(args.model_save_path)
    
    command_save_filename = os.path.join(args.model_save_path, "command.txt")
    with open(command_save_filename, "w") as f:
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        if CUDA_VISIBLE_DEVICES is not None:
            f.write(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python3 {' '.join(sys.argv)}")
        else:
            f.write(f"python3 {' '.join(sys.argv)}")
    
    args_save_filename = os.path.join(args.model_save_path, "args.json")
    with open(args_save_filename, "w") as f:
        json.dump(args.__dict__, f, indent=4, ensure_ascii=False)
    
    logging_file = os.path.join(args.model_save_path, "stdout.log")
    logging_stream = open(logging_file, "a")
    logging.basicConfig(logging_stream)
    
    logging.info(f"Training arguments:\n{json.dumps(args.__dict__, indent=4, ensure_ascii=False)}")
    logging.info(f"Run experiments with random seed: {args.seed}")
    logging.info(f"Saved training command to file: {command_save_filename}")
    logging.info(f"Saved training arguments to file: {args_save_filename}")

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
        if args.task == "kbqa_name_expanded_pairwise" and not paths.strip():
            continue
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

def main(args):
    check_args(args)
    prepare_for_training(args)
    
    train_filename = os.path.join(args.data_path, args.train_filename)
    dev_filename = os.path.join(args.data_path, args.dev_filename)
    test_filename = os.path.join(args.data_path, args.test_filename)
    
    if args.task == "kbqa":
        train_loader = OriginalWebQSPDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True)
        dev_loader = OriginalWebQSPDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False)
        test_loader = OriginalWebQSPDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False)
    elif args.task == "kbqa_name_expanded":
        train_loader = NameExpandedWebQSPDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, separate_token=args.separate_token, annotate_self_token=args.annotate_self_token, use_prompt=args.use_prompt, mix_webred_data=args.mix_webred_data, mix_webred_matching_data=args.mix_webred_matching_data, mix_bertrl_data=args.mix_bertrl_data, bertrl_max_samples=args.bertrl_max_samples)
        dev_loader = NameExpandedWebQSPDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, separate_token=args.separate_token, annotate_self_token=args.annotate_self_token, use_prompt=args.use_prompt, mix_webred_data=args.mix_webred_data, mix_webred_matching_data=args.mix_webred_matching_data, mix_bertrl_data=args.mix_bertrl_data, bertrl_max_samples=args.bertrl_max_samples)
        test_loader = NameExpandedWebQSPDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, separate_token=args.separate_token, annotate_self_token=args.annotate_self_token, use_prompt=args.use_prompt, mix_webred_data=args.mix_webred_data, mix_webred_matching_data=args.mix_webred_matching_data, mix_bertrl_data=args.mix_bertrl_data, bertrl_max_samples=args.bertrl_max_samples)
    elif args.task == "kbqa_name_expanded_pairwise":
        train_loader = PWNameExpandedWebQSPDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, separate_token=args.separate_token, annotate_self_token=args.annotate_self_token, use_prompt=args.use_prompt, mix_webred_data=args.mix_webred_data, mix_webred_matching_data=args.mix_webred_matching_data, mix_bertrl_data=args.mix_bertrl_data)
        dev_loader = NameExpandedWebQSPDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, separate_token=args.separate_token, annotate_self_token=args.annotate_self_token, use_prompt=args.use_prompt, mix_webred_data=args.mix_webred_data, mix_webred_matching_data=args.mix_webred_matching_data, mix_bertrl_data=args.mix_bertrl_data)
        test_loader = NameExpandedWebQSPDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, separate_token=args.separate_token, annotate_self_token=args.annotate_self_token, use_prompt=args.use_prompt, mix_webred_data=args.mix_webred_data, mix_webred_matching_data=args.mix_webred_matching_data, mix_bertrl_data=args.mix_bertrl_data)
    elif args.task == "kbqa_multi_paths":
        train_loader = SplitedPathWebQSPDataLoader(train_filename, "train", args.model_name_or_path, max_batch_size=args.batch_size, shuffle=True, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, use_aggregate=args.use_aggregate)
        dev_loader = SplitedPathWebQSPDataLoader(dev_filename, "dev", args.model_name_or_path, max_batch_size=args.batch_size, shuffle=False, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, use_aggregate=args.use_aggregate)
        test_loader = SplitedPathWebQSPDataLoader(test_filename, "test", args.model_name_or_path, max_batch_size=args.batch_size, shuffle=False, entity_start_token=args.entity_start_token, entity_end_token=args.entity_end_token, head_entity_start_token=args.head_entity_start_token, head_entity_end_token=args.head_entity_end_token, tail_entity_start_token=args.tail_entity_start_token, tail_entity_end_token=args.tail_entity_end_token, unknown_entity_token=args.unknown_entity_token, use_aggregate=args.use_aggregate)
    elif args.task == "webred":
        train_loader = WebREDForPretrainingDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True)
        dev_loader = WebREDForPretrainingDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False)
        test_loader = WebREDForPretrainingDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False)
    elif args.task == "kbqa_with_prompt":
        train_loader = OriginalWebQSPWithPromptDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True)
        dev_loader = OriginalWebQSPWithPromptDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False)
        test_loader = OriginalWebQSPWithPromptDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False)
    elif args.task == "mixed_kbqa":
        train_loader = MixedWebQSPWebREDDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True)
        dev_loader = MixedWebQSPWebREDDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False)
        test_loader = MixedWebQSPWebREDDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False)
    
    model = BertForKBQA(args.model_name_or_path, dropout=args.dropout).cuda()
    if args.continue_training_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.continue_training_path, "pytorch_model.bin")), strict=False)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)
    tensorboard_writer = SummaryWriter(args.tensorboard_log_dir or os.path.join(args.model_save_path, "logs"))
    
    if args.use_apex_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_amp_opt_level)
    
    logging.info(f"Start training, number of epochs: {args.num_epochs}")
    global_step = 0
    best_dev_performance = 0.0
    for epoch_idx in range(args.num_epochs):
        logging.info(f"Training start for epoch {epoch_idx}")
        
        sum_train_loss = 0.0
        sum_train_acc = 0.0
        num_batches = math.ceil(train_loader.num_data / args.batch_size)
        logging.info(f"Total number of batches: {num_batches}")
        with tqdm.trange(num_batches) as trange_obj:
            for batch_idx in trange_obj:
                trange_obj.set_postfix(loss=f"{sum_train_loss / batch_idx if batch_idx > 0 else 0.0:10.6f}", acc=f"{sum_train_acc / batch_idx if batch_idx > 0 else 0.0:10.6f}")
                
                model.train()
                batch = train_loader.get_batch(batch_idx, args.batch_size, args.max_seq_length)
                batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]
                if args.task == "kbqa_multi_paths" and args.use_aggregate:
                    input_ids, token_type_ids, attention_mask, offsets, labels, q_ids, candidates = batch
                    logits = model(input_ids, token_type_ids, attention_mask)
                    max_logits = []
                    for offset_idx in range(len(labels)):
                        if offsets[offset_idx] == offsets[offset_idx + 1]:
                            max_logits.append(torch.tensor(-1e6).cuda())
                        else:
                            max_logits.append(torch.max(logits[offsets[offset_idx]:offsets[offset_idx + 1]]))
                    logits = torch.stack(max_logits)
                    preds = (torch.sigmoid(logits) > 0.5).long()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
                    acc = cal_accuracy(preds, labels)
                elif args.task == "kbqa_name_expanded_pairwise":
                    input_ids_pos, token_type_ids_pos, attention_mask_pos, input_ids_neg, token_type_ids_neg, attention_mask_neg, q_ids, candidates = batch
                    logits_pos = model(input_ids_pos, token_type_ids_pos, attention_mask_pos)
                    logits_neg = model(input_ids_neg, token_type_ids_neg, attention_mask_neg)
                    score_pos = torch.sigmoid(logits_pos)
                    score_neg = torch.sigmoid(logits_neg)
                    loss = torch.max(torch.zeros(score_pos.shape).cuda().half(), args.margin-(score_pos-score_neg))
                    loss = loss.mean()
                    preds_pos = (torch.sigmoid(logits_pos) > 0.5).long()
                    preds_neg = (torch.sigmoid(logits_neg) > 0.5).long()
                    acc = (cal_accuracy(preds_pos, torch.ones(score_pos.shape).cuda().half())+cal_accuracy(preds_neg, torch.zeros(score_pos.shape).cuda().half()))/2
                else:
                    input_ids, token_type_ids, attention_mask, labels, q_ids, candidates = batch

                    logits = model(input_ids, token_type_ids, attention_mask)
                
                    preds = (torch.sigmoid(logits) > 0.5).long()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
                    acc = cal_accuracy(preds, labels)
                    
                sum_train_acc += acc
                sum_train_loss += loss.item()
                tensorboard_writer.add_scalar("train_loss", loss.item(), global_step=global_step)
                tensorboard_writer.add_scalar("train_acc", acc, global_step=global_step)

                model.zero_grad()
                optimizer.zero_grad()

                if args.use_apex_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss_value:
                        scaled_loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.gradient_clip)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

                optimizer.step()
                global_step += 1

                if (batch_idx + 1) % args.evaluation_steps == 0 or batch_idx == num_batches - 1:
                    logging.info(f"Evaluating on dev dataset at step {batch_idx} in epoch {epoch_idx} (global step: {global_step})...")
                    
                    if args.metric in ("hits", "f1"):
                        threshold, p, r, f1, hits, loss = evaluate_kbqa(args, dev_loader, model, args.batch_size, args.max_seq_length, thresholds=args.threshold, log_file=os.path.join(args.model_save_path, f"evaluate_dev_step{global_step}.json"), multi_paths=(args.task == "kbqa_multi_paths" and args.use_aggregate))
                        logging.info(f"Results: f1 = {f1:.6f} (threshold = {threshold:.6f}), hits@1 = {hits:.6f}")
                        tensorboard_writer.add_scalar("dev_precision", p, global_step=global_step)
                        tensorboard_writer.add_scalar("dev_recall", r, global_step=global_step)
                        tensorboard_writer.add_scalar("dev_f1", f1, global_step=global_step)
                        tensorboard_writer.add_scalar("dev_hits", hits, global_step=global_step)
                        tensorboard_writer.add_scalar("dev_loss", loss, global_step=global_step)
                        if args.metric == "hits":
                            current_dev_performance = hits
                        else:
                            current_dev_performance = f1
                    elif args.metric == "accuracy":
                        acc = evaluate_accuracy(dev_loader, model, args.batch_size, args.max_seq_length, log_file=os.path.join(args.model_save_path, f"evaluate_dev_step{global_step}.json"))
                        logging.info(f"Results: acc = {acc:.6f}")
                        tensorboard_writer.add_scalar("dev_accuracy", acc, global_step=global_step)
                        current_dev_performance = acc
                    
                    # selecting best performed model
                    if current_dev_performance > best_dev_performance:
                        model_save_file = os.path.join(args.model_save_path, "pytorch_model.bin")
                        logging.info(f"Best performance achieved, save model to {model_save_file}")
                        torch.save(model.state_dict(), model_save_file)
                        best_dev_performance = current_dev_performance
                        if args.metric in ("hits", "f1"):
                            best_dev_threshold = threshold

    # evaluate on test dataset
    logging.info("Evaluation on test dataset...")
    model.load_state_dict(torch.load(os.path.join(args.model_save_path, "pytorch_model.bin")))
    model.cuda()
    if args.metric in ("hits", "f1"):
        threshold, p, r, f1, hits, loss = evaluate_kbqa(args, test_loader, model, args.batch_size, args.max_seq_length, thresholds=best_dev_threshold, log_file=os.path.join(args.model_save_path, "evaluate_test.json"), multi_paths=(args.task == "kbqa_multi_paths" and args.use_aggregate))
        logging.info(f"Results: f1 = {f1:.6f} (threshold = {threshold:.6f}), hits@1 = {hits:.6f}")
    elif args.metric == "accuracy":
        acc = evaluate_accuracy(test_loader, model, args.batch_size, args.max_seq_length, log_file=os.path.join(args.model_save_path, "evaluate_test.json"))
        logging.info(f"Results: acc = {acc:.6f}")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)