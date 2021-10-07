import os
import sys
import tqdm
import json
import random
import collections

def add_processed_data(processed_data, item1, item2):
    processed_data.append({
        "text1": item1["annotated_sentence_text"],
        "text2": item2["annotated_sentence_text"],
        "relation1": item1["relation_name"],
        "relation2": item2["relation_name"],
        "label": int(item1["relation_name"] == item2["relation_name"])
    })
    
def sample_for_single_split(input_file, output_file, num_neg=9):
    data = json.load(open(input_file))
    relation_indexed_sample = collections.defaultdict(list)
    for item in data:
        if item["num_pos_raters"] <= item["num_raters"] - item["num_pos_raters"]:
            continue
        relation_name = item["relation_name"]
        relation_indexed_sample[relation_name].append(item)
        
    processed_data = []
    for item in tqdm.tqdm(data):
        if item["num_pos_raters"] <= item["num_raters"] - item["num_pos_raters"]:
            continue
        text = item["annotated_sentence_text"]
        relation_name = item["relation_name"]
        all_positive_samples = []
        for each in relation_indexed_sample[relation_name]:
            if each["annotated_sentence_text"] == text:
                continue
            all_positive_samples.append(each)
        if len(all_positive_samples) == 0:
            continue
        pos = random.choice(all_positive_samples)
        all_negative_samples = []
        for each in data:
            if item["num_pos_raters"] <= item["num_raters"] - item["num_pos_raters"]:
                continue
            if each["relation_name"] != relation_name:
                all_negative_samples.append(each)
        random.shuffle(all_negative_samples)
        negs = all_negative_samples[:num_neg]
        add_processed_data(processed_data, item, pos)
        for neg in negs:
            add_processed_data(processed_data, item, neg)
    with open(output_file, "w") as f_out:
        json.dump(processed_data, f_out)


def main():
    train_input = "../webred/webred_21.json"
    test_input = "../webred/webred_5.json"
    train_output = "./webred_21_for_matching.json"
    test_output = "./webred_5_for_matching.json"
    sample_for_single_split(test_input, test_output)
    sample_for_single_split(train_input, train_output)


if __name__ == "__main__":
    main()