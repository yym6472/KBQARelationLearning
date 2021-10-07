import json
import tqdm
import random
import collections
import tensorflow as tf

def read_examples(*dataset_paths):
    examples = []
    dataset = tf.data.TFRecordDataset(dataset_paths)
    for raw_sentence in dataset:
        sentence = tf.train.Example()
        sentence.ParseFromString(raw_sentence.numpy())
        examples.append(sentence)
    return examples

def get_feature(sentence, feature_name, idx=0):
  feature = sentence.features.feature[feature_name]
  return getattr(feature, feature.WhichOneof('kind')).value[idx]

def tfrecord2json(input_file, output_file):
    examples = read_examples(input_file)
    converted_example_list = []
    for sentence in tqdm.tqdm(examples):
        annotated_sentence_text = get_feature(sentence, 'sentence').decode('utf-8')
        relation_name = get_feature(sentence, 'relation_name').decode('utf-8')
        relation_id = get_feature(sentence, 'relation_id').decode('utf-8')
        source_name = get_feature(sentence, 'source_name').decode('utf-8')
        target_name = get_feature(sentence, 'target_name').decode('utf-8')
        num_pos_raters = get_feature(sentence, 'num_pos_raters')
        num_raters = get_feature(sentence, 'num_raters')
        assert isinstance(annotated_sentence_text, str) and annotated_sentence_text.strip()
        assert isinstance(relation_name, str) and relation_name.strip()
        assert isinstance(relation_id, str) and relation_id.strip()
        assert isinstance(source_name, str) and source_name.strip()
        assert isinstance(target_name, str) and target_name.strip()
        assert isinstance(num_pos_raters, int) and num_pos_raters >= 0
        assert isinstance(num_raters, int) and num_raters > 0
        converted_example_list.append({
            'annotated_sentence_text': annotated_sentence_text,
            'relation_name': relation_name,
            'relation_id': relation_id,
            'source_name': source_name,
            'target_name': target_name,
            'num_pos_raters': num_pos_raters,
            'num_raters': num_raters
        })
    json.dump(converted_example_list, open(output_file, 'w'))

def prepare_data_for_pretraining(input_file, output_file, split, chain_recorder):
    examples = json.load(open(input_file, 'r'))
    converted_example_list = []
    for idx, example in tqdm.tqdm(enumerate(examples)):
        if example['num_pos_raters'] == example['num_raters'] - example['num_pos_raters']:
            continue
        path = example['source_name'] + '~' + example['relation_name'] + '~' + example['target_name']
        example_id = f'WebRED{split}-{idx}'
        label = 0 if example['num_pos_raters'] < example['num_raters'] - example['num_pos_raters'] else 1
        example_obj = {
            'q': example['annotated_sentence_text'].replace('^', ' ').replace('&', ' ').replace('$', ' ').strip(),
            'path': [path],
            'id': example_id,
            'candi': '',
            'label': label
        }
        if label == 0:
            converted_example_list.append({
                "true": [],
                "false": [example_obj]
            })
        else:
            converted_example_list.append({
                "true": [example_obj],
                "false": []
            })
        chain_recorder[example_id] = [example['relation_name']]
    json.dump(converted_example_list, open(output_file, 'w'))
    
def prepare_data_for_pretraining_with_relation_mapped(input_file, output_file, split, chain_recorder):
    
    # prepare resources for mapping
    with open("./wikidata_relations_with_entry_in_freebase.txt", "r") as f:
        relation_ids = [line.strip().split("\t")[1] for line in f]
    assert len(relation_ids) > 0 and all(r_id[0] == "P" for r_id in relation_ids)
    freebase_relation_names = []
    with open("./wikidata_to_freebase_relations_mapping.txt", "r") as f:
        for line in f:
            top1_relation = line.strip().split("\t")[0]
            if top1_relation.startswith("https://www.freebase.com/"):
                freebase_relation_names.append(top1_relation[25:].replace("/", " ").replace("_", " ").strip())
            elif top1_relation.startswith("http://www.freebase.com/"):
                freebase_relation_names.append(top1_relation[24:].replace("/", " ").replace("_", " ").strip())
            else:
                raise ValueError("Relation names error")
    assert len(relation_ids) == len(freebase_relation_names)
    wikidata_id_to_freebase_name = {k: v for k, v in zip(relation_ids, freebase_relation_names)}
    
    examples = json.load(open(input_file, 'r'))
    converted_example_list = []
    for idx, example in tqdm.tqdm(enumerate(examples)):
        if example['num_pos_raters'] == example['num_raters'] - example['num_pos_raters']:
            continue
        if example['relation_id'] in wikidata_id_to_freebase_name:
            path = example['source_name'] + '~' + wikidata_id_to_freebase_name[example['relation_id']] + '~' + example['target_name']
        else:
            path = example['source_name'] + '~' + example['relation_name'] + '~' + example['target_name']
        example_id = f'WebRED{split}-{idx}'
        label = 0 if example['num_pos_raters'] < example['num_raters'] - example['num_pos_raters'] else 1
        example_obj = {
            'q': example['annotated_sentence_text'].replace('^', ' ').replace('&', ' ').replace('$', ' ').strip(),
            'path': [path],
            'id': example_id,
            'candi': '',
            'label': label
        }
        if label == 0:
            converted_example_list.append({
                "true": [],
                "false": [example_obj]
            })
        else:
            converted_example_list.append({
                "true": [example_obj],
                "false": []
            })
        chain_recorder[example_id] = [example['relation_name']]
    json.dump(converted_example_list, open(output_file, 'w'))

def merge_two_examples(example1, example2, split, example1_idx):
    text1 = example1['annotated_sentence_text'].replace('^', ' ').replace('&', ' ').replace('$', ' ').strip()
    text2 = example2['annotated_sentence_text'].replace('^', ' ').replace('&', ' ').replace('$', ' ').strip()
    text = text1 + "; " + text2
    path = "~".join([example1["source_name"], example1["relation_name"], example1["target_name"], example2["relation_name"], example2["target_name"]])
    example_id = f"WebRED{split}-{example1_idx} & {example2['id']}"
    label1 = 0 if example1['num_pos_raters'] < example1['num_raters'] - example1['num_pos_raters'] else 1
    label2 = 0 if example2['num_pos_raters'] < example2['num_raters'] - example2['num_pos_raters'] else 1
    label = 1 if label1 == 1 and label2 == 1 else 0
    example_obj = {
        "q": text,
        "path": [path],
        "id": example_id,
        "candi": "",
        "label": label
    }
    if example_obj["label"] == 0:
        return {
            "true": [],
            "false": [example_obj]
        }
    else:
        return {
            "true": [example_obj],
            "false": []
        }
    
def prepare_data_for_pretraining_2hop(input_file, output_file, split, chain_recorder):
    examples = json.load(open(input_file, 'r'))
    converted_example_list = []
    
    print("Adding 1-hop samples")
    num_added = 0
    for idx, example in tqdm.tqdm(enumerate(examples)):
        if example['num_pos_raters'] == example['num_raters'] - example['num_pos_raters']:
            continue
        path = example['source_name'] + '~' + example['relation_name'] + '~' + example['target_name']
        example_id = f'WebRED{split}-{idx}'
        label = 0 if example['num_pos_raters'] < example['num_raters'] - example['num_pos_raters'] else 1
        example_obj = {
            'q': example['annotated_sentence_text'].replace('^', ' ').replace('&', ' ').replace('$', ' ').strip(),
            'path': [path],
            'id': example_id,
            'candi': '',
            'label': label
        }
        if label == 0:
            converted_example_list.append({
                "true": [],
                "false": [example_obj]
            })
        else:
            converted_example_list.append({
                "true": [example_obj],
                "false": []
            })
        num_added += 1
        chain_recorder[example_id] = [example['relation_name']]
    print(f"Total {num_added} examples added")
    
    print("Adding 2-hop samples")
    num_added = 0
    head_entity_indexed_paths = collections.defaultdict(list)
    for idx, example in enumerate(examples):
        if example['num_pos_raters'] == example['num_raters'] - example['num_pos_raters']:
            continue
        head_entity = example["source_name"]
        obj = {}
        obj.update(example)
        obj["label"] = int(example['num_pos_raters'] > example['num_raters'] - example['num_pos_raters'])
        obj["id"] = f"WebRED{split}-{idx}"
        head_entity_indexed_paths[head_entity].append(obj)
        
    for idx, example in tqdm.tqdm(enumerate(examples)):
        if example['num_pos_raters'] == example['num_raters'] - example['num_pos_raters']:
            continue
        this_tail_entity = example["target_name"]
        this_label = int(example['num_pos_raters'] > example['num_raters'] - example['num_pos_raters'])
        positive_paths = [item for item in head_entity_indexed_paths[this_tail_entity] if item["label"] == 1]
        negative_paths = [item for item in head_entity_indexed_paths[this_tail_entity] if item["label"] == 0]
        
        if this_label == 0:
            if len(positive_paths) > 0:
                positive_example = random.sample(positive_paths, 1)[0]
                converted_example_list.append(merge_two_examples(example, positive_example, split, idx))
                num_added += 1
        else:
            if len(positive_paths) > 0:
                positive_example = random.sample(positive_paths, 1)[0]
                converted_example_list.append(merge_two_examples(example, positive_example, split, idx))
                num_added += 1
            if len(negative_paths) > 0:
                negative_example = random.sample(negative_paths, 1)[0]
                converted_example_list.append(merge_two_examples(example, negative_example, split, idx))
                num_added += 1
    print(f"Total {num_added} examples added")
    
    json.dump(converted_example_list, open(output_file, 'w'))
    
def update_chain_info(chain_recorder, input_file, output_file):
    others = json.load(open(input_file, 'r'))
    chain_recorder.update(others)
    json.dump(chain_recorder, open(output_file, 'w'))

def record_all_relations(input_file_1, input_file_2, output_file):
    data_1 = json.load(open(input_file_1))
    data_2 = json.load(open(input_file_2))
    data = data_1 + data_2
    all_relations = set()
    name2id = {}
    name2freq = collections.defaultdict(int)
    for item in data:
        relation_name = item["relation_name"]
        relation_id = item["relation_id"]
        if relation_name not in name2id:
            name2id[relation_name] = relation_id
        assert relation_id == name2id[relation_name]
        all_relations.add(relation_name)
        name2freq[relation_name] += 1
    with open(output_file, "w") as f_out:
        all_relations = sorted(list(all_relations))
        for relation in all_relations:
            f_out.write(f"{relation}\t{name2id[relation]}\t{name2freq[relation]}\n")

def count_occurrence(s, key):
    count = 0
    idx = s.find(key, 0)
    while idx != -1:
        count += 1
        idx = s.find(key, idx + len(key))
    return count

def find_mapping_relations(input_file, mapping_html_file, output_file):
    """
    寻找Wikidata中的哪些关系在freebase中有对应的。
    """
    with open(mapping_html_file, "r") as f:
        html = f.read()
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f]
    results = []
    for line in lines:
        _, r_id, _ = line.split("\t")
        if f"({r_id})" in html:
            occ = count_occurrence(html, f"({r_id})")
            results.append(line + f"\t{occ}")
    with open(output_file, "w") as f:
        for item in results:
            f.write(item + "\n")
        
            
if __name__ == '__main__':
#     tfrecord2json('./WebRED/webred_21.tfrecord', './webred_21.json')
#     tfrecord2json('./WebRED/webred_5.tfrecord', './webred_5.json')

#     chain_recorder = {}
#     prepare_data_for_pretraining('./webred_21.json', './webred_21_pretraining.json', '21', chain_recorder)
#     prepare_data_for_pretraining('./webred_5.json', './webred_5_pretraining.json', '5', chain_recorder)
#     update_chain_info(chain_recorder, '../../../K-BERT/full/all_chains.json', './all_chains.json')

#     record_all_relations("./webred_21.json", "./webred_5.json", "./relations.txt")
#     find_mapping_relations("./relations.txt", "./freebase_to_wikidata_html.txt", "./wikidata_relations_with_entry_in_freebase.txt")
#     prepare_data_for_pretraining_with_relation_mapped('./webred_21.json', './webred_21_pretraining_with_relation_mapped.json', '21', {})
#     prepare_data_for_pretraining_with_relation_mapped('./webred_5.json', './webred_5_pretraining_with_relation_mapped.json', '5', {})
    prepare_data_for_pretraining_2hop('./webred_21.json', './webred_21_pretraining_2hop.json', '21', {})
    prepare_data_for_pretraining_2hop('./webred_5.json', './webred_5_pretraining_2hop.json', '5', {})