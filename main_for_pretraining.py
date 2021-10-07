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
from data_loader import WebREDMatchingDataLoader, WebREDMaskedTokenPredictionDataLoader, BERTRLMatchingDataLoader, PWWebREDMatchingDataLoader, JointMatchingDataLoader
from model import BertForPretraining


def parse_args():
    """
    Argument settings.
    """
    # arguments for experiments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducing experimental results")
    parser.add_argument("--margin", type=float, required=False, default=0.3, help="Pairwise loss margin")
    parser.add_argument("--task", type=str, required=True, choices=["matching", "masked_token_prediction", "joint", "matching_pw"], help="Define the type of training task (or dataset)")
    parser.add_argument("--model_name_or_path", type=str, default="./bert-base-uncased/", help="The model path or model name of pre-trained model")
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
#     parser.add_argument("--threshold", type=float, default=None, help="The threshold for binary classification. If set to None, it will be automatically inferred from the dev set")
#     parser.add_argument("--entity_start_token", type=str, default=None, help="The token to annotate the start of an entity")
#     parser.add_argument("--entity_end_token", type=str, default=None, help="The token to annotate the end of an entity")
#     parser.add_argument("--head_entity_start_token", type=str, default=None, help="The token to annotate the start of a head entity")
#     parser.add_argument("--head_entity_end_token", type=str, default=None, help="The token to annotate the end of a head entity")
#     parser.add_argument("--tail_entity_start_token", type=str, default=None, help="The token to annotate the start of a tail entity")
#     parser.add_argument("--tail_entity_end_token", type=str, default=None, help="The token to annotate the end of a tail entity")
#     parser.add_argument("--unknown_entity_token", type=str, default="[unknown entity]", help="The token to annotate entities with no entity name found")
#     parser.add_argument("--separate_token", type=str, default="[path separator]", help="The token to separator multiple paths in the combined paths")
#     parser.add_argument("--annotate_self_token", type=str, default=None, help="The token to annotate that the candidate entity is equal to the topic entity")
#     parser.add_argument("--use_aggregate", action="store_true", help="Use path aggregate when training")
#     parser.add_argument("--use_prompt", action="store_true", help="Use prompt for question when training KBQA")
#     parser.add_argument("--mix_webred_data", action="store_true", help="Mix WebRED data when training KBQA")

    # arguments for training (or data preprocessing)
    parser.add_argument("--batch_size", type=int, default=128, help="Training mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="The learning rate")
    parser.add_argument("--evaluation_steps", type=int, default=1000, help="The steps between every evaluations")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The max sequence length")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="The max norm of gradient to apply gradient clip")
    parser.add_argument("--patience", type=int, default=None, help="Patience for early stop")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "token_accuracy"], help="The metric to select the best model")
    
    return parser.parse_args()

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

def cal_token_accuracy(preds, labels):
    """
    Calculate the accuracy for masked token prediction (only used during the training phase).
    """
    num_correct = ((preds == labels) & (labels != -100)).int().sum().item()
    num_total = (labels != -100).int().sum().item()
    return num_correct / num_total

def evaluate_accuracy(args, data_loader, model, batch_size, max_seq_length, log_file=None):
    """
    Calculate accuracy for evaluation.
    """ 
    # evaluate by iterating data loader
    all_scores, all_labels, losses = [], [], []
    model.eval()
    for batch_idx in tqdm.tqdm(range(math.ceil(data_loader.num_data / batch_size))):
        batch = data_loader.get_batch(batch_idx, batch_size, max_seq_length)
        batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]
        if args.task == "matching" or args.task == "joint":
            input_ids, token_type_ids, attention_mask, labels = batch

            logits, _ = model(input_ids, token_type_ids, attention_mask)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            all_scores.extend(torch.sigmoid(logits).tolist())
            all_labels.extend(labels.tolist())
            losses.append(loss.item())
        elif args.task == "matching_pw":
            input_ids_pos, token_type_ids_pos, attention_mask_pos, input_ids_neg, token_type_ids_neg, attention_mask_neg = batch
            logits_pos, _ = model(input_ids_pos, token_type_ids_pos, attention_mask_pos)
            logits_neg, _ = model(input_ids_neg, token_type_ids_neg, attention_mask_neg)
            score_pos = torch.sigmoid(logits_pos)
            score_neg = torch.sigmoid(logits_neg)
            if args.margin is None: args.margin = 0.3
            loss = torch.max(torch.zeros(score_pos.shape).cuda().half(), args.margin-(score_pos-score_neg))
            loss = loss.mean()
            all_scores.extend(score_pos.tolist())
            all_labels.extend([1] * len(score_pos))
            all_scores.extend(score_neg.tolist())
            all_labels.extend([0] * len(score_neg))
            losses.append(loss.item())
        else:
            raise ValueError("Invalid task name with sentence level accuracy")
    
    # calculate accuracy
    if args.task == "matching_pw":
        question_wise_scores = []
        for score, label in zip(all_scores, all_labels):
            question_wise_scores.append({
                "label": label,
                "score": score,
                "accuracy": int(int(score > 0.5) == label)
            })
    else:
        all_samples = data_loader.data
        question_wise_scores = []
        for score, label, sample in zip(all_scores, all_labels, all_samples):
            if isinstance(sample, str):
                _, _, _, text1, text2 = sample.strip().split("\t")
                question_wise_scores.append({
                    "text1": text1,
                    "text2": text2,
                    "label": label,
                    "score": score,
                    "accuracy": int(int(score > 0.5) == label)
                })
            elif "text1" in sample:
                question_wise_scores.append({
                    "text1": sample["text1"],
                    "text2": sample["text2"],
                    "label": label,
                    "score": score,
                    "accuracy": int(int(score > 0.5) == label)
                })
            else:
                question_wise_scores.append({
                    "text1": sample["question"],
                    "text2": sample["paths"],
                    "label": label,
                    "score": score,
                    "accuracy": int(int(score > 0.5) == label)
                })
                

    all_acc = [item["accuracy"] for item in question_wise_scores]
    overall = {
        "accuracy": sum(all_acc) / len(all_acc)
    }
    
    # log and return
    if log_file is not None:
        json.dump(overall, open(log_file[:-5] + ".overall.json", "w"), indent=4)
        json.dump(question_wise_scores, open(log_file, "w"), indent=4)
        
    return overall["accuracy"]

def evaluate_token_accuracy(data_loader, model, batch_size, max_seq_length, log_file=None):
    """
    Calculate accuracy for evaluation.
    """ 
    # evaluate by iterating data loader
    num_correct, num_total, losses = 0, 0, []
    model.eval()
    for batch_idx in tqdm.tqdm(range(math.ceil(data_loader.num_data / batch_size))):
        batch = data_loader.get_batch(batch_idx, batch_size, max_seq_length)
        batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]
        input_ids, token_type_ids, attention_mask, labels = batch
        _, logits = model(input_ids, token_type_ids, attention_mask)
        preds = logits.argmax(-1)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        acc = cal_token_accuracy(preds, labels)
        num_correct += ((preds == labels) & (labels != -100)).int().sum().item()
        num_total += (labels != -100).int().sum().item()
        losses.append(loss.item())
    
    overall = {
        "accuracy": num_correct / num_total,
        "loss": sum(losses) / len(losses)
    }
    
    # log and return
    if log_file is not None:
        json.dump(overall, open(log_file[:-5] + ".overall.json", "w"), indent=4)
        
    return overall["accuracy"]

def main(args):
    prepare_for_training(args)
    
    train_filename = os.path.join(args.data_path, args.train_filename)
    dev_filename = os.path.join(args.data_path, args.dev_filename)
    test_filename = os.path.join(args.data_path, args.test_filename)
    
    if args.task == "matching":
        if "bertrl" in args.data_path:
            train_loader = BERTRLMatchingDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True)
            dev_loader = BERTRLMatchingDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False)
            test_loader = BERTRLMatchingDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False)
        else:
            train_loader = WebREDMatchingDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True)
            dev_loader = WebREDMatchingDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False)
            test_loader = WebREDMatchingDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False)
    elif args.task == "masked_token_prediction":
        train_loader = WebREDMaskedTokenPredictionDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True)
        dev_loader = WebREDMaskedTokenPredictionDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False)
        test_loader = WebREDMaskedTokenPredictionDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False)
    elif args.task == "joint":
        train_loader = JointMatchingDataLoader("train", args.model_name_or_path, shuffle=True, bertrl_max_samples=100000)
        dev_loader = JointMatchingDataLoader("dev", args.model_name_or_path, shuffle=False, bertrl_max_samples=1000)
        test_loader = JointMatchingDataLoader("test", args.model_name_or_path, shuffle=False, bertrl_max_samples=1000)
        pass
    elif args.task == "matching_pw":
        train_loader = PWWebREDMatchingDataLoader(train_filename, "train", args.model_name_or_path, shuffle=True)
        dev_loader = PWWebREDMatchingDataLoader(dev_filename, "dev", args.model_name_or_path, shuffle=False)
        test_loader = PWWebREDMatchingDataLoader(test_filename, "test", args.model_name_or_path, shuffle=False)
        pass
    
    model = BertForPretraining(args.task, args.model_name_or_path, vocab_size=30522, dropout=args.dropout).cuda()
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
        with tqdm.trange(num_batches) as trange_obj:
            for batch_idx in trange_obj:
                trange_obj.set_postfix(loss=f"{sum_train_loss / batch_idx if batch_idx > 0 else 0.0:10.6f}", acc=f"{sum_train_acc / batch_idx if batch_idx > 0 else 0.0:10.6f}")
                
                model.train()
                batch = train_loader.get_batch(batch_idx, args.batch_size, args.max_seq_length)
                batch = [item.cuda() if isinstance(item, torch.Tensor) else item for item in batch]
                if args.task == "matching" or args.task == "joint":
                    input_ids, token_type_ids, attention_mask, labels = batch
                    logits, _ = model(input_ids, token_type_ids, attention_mask)
                    preds = (torch.sigmoid(logits) > 0.5).long()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
                    acc = cal_accuracy(preds, labels)
                elif args.task == "masked_token_prediction":
                    input_ids, token_type_ids, attention_mask, labels = batch
                    _, logits = model(input_ids, token_type_ids, attention_mask)
                    preds = logits.argmax(-1)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
                    acc = cal_token_accuracy(preds, labels)
                elif args.task == "matching_pw":
                    input_ids_pos, token_type_ids_pos, attention_mask_pos, input_ids_neg, token_type_ids_neg, attention_mask_neg = batch
                    logits_pos, _ = model(input_ids_pos, token_type_ids_pos, attention_mask_pos)
                    logits_neg, _ = model(input_ids_neg, token_type_ids_neg, attention_mask_neg)
                    score_pos = torch.sigmoid(logits_pos)
                    score_neg = torch.sigmoid(logits_neg)
                    if args.margin is None: args.margin = 0.3
                    loss = torch.max(torch.zeros(score_pos.shape).cuda().half(), args.margin-(score_pos-score_neg))
                    loss = loss.mean()
                    preds_pos = (torch.sigmoid(logits_pos) > 0.5).long()
                    preds_neg = (torch.sigmoid(logits_neg) > 0.5).long()
                    acc = (cal_accuracy(preds_pos, torch.ones(score_pos.shape).cuda().half())+cal_accuracy(preds_neg, torch.zeros(score_pos.shape).cuda().half()))/2
                    
                
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
                    
                    if args.metric == "accuracy":
                        acc = evaluate_accuracy(args, dev_loader, model, args.batch_size, args.max_seq_length, log_file=os.path.join(args.model_save_path, f"evaluate_dev_step{global_step}.json"))
                        logging.info(f"Results: acc = {acc:.6f}")
                        tensorboard_writer.add_scalar("dev_accuracy", acc, global_step=global_step)
                        current_dev_performance = acc
                    elif args.metric == "token_accuracy":
                        acc = evaluate_token_accuracy(dev_loader, model, args.batch_size, args.max_seq_length, log_file=os.path.join(args.model_save_path, f"evaluate_dev_step{global_step}.json"))
                        logging.info(f"Results: acc = {acc:.6f}")
                        tensorboard_writer.add_scalar("dev_accuracy", acc, global_step=global_step)
                        current_dev_performance = acc
                    
                    # selecting best performed model
                    if current_dev_performance > best_dev_performance:
                        model_save_file = os.path.join(args.model_save_path, "pytorch_model.bin")
                        logging.info(f"Best performance achieved, save model to {model_save_file}")
                        torch.save(model.state_dict(), model_save_file)
                        best_dev_performance = current_dev_performance

    # evaluate on test dataset
    logging.info("Evaluation on test dataset...")
    model.load_state_dict(torch.load(os.path.join(args.model_save_path, "pytorch_model.bin")))
    model.cuda()
    if args.metric == "accuracy":
        acc = evaluate_accuracy(args, test_loader, model, args.batch_size, args.max_seq_length, log_file=os.path.join(args.model_save_path, "evaluate_test.json"))
        logging.info(f"Results: acc = {acc:.6f}")
    elif args.metric == "token_accuracy":
        acc = evaluate_token_accuracy(test_loader, model, args.batch_size, args.max_seq_length, log_file=os.path.join(args.model_save_path, "evaluate_test.json"))

if __name__ == "__main__":
    args = parse_args()
    main(args)