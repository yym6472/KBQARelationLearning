import json
import random
import tqdm
import collections


def is_freebase_mid(kb_id):
    return kb_id.startswith("<fb:") and kb_id.endswith(">")

def reversed_tuple(_tuple):
    return (_tuple[2], _tuple[1], _tuple[0])

def convert_entity(entity, mid2name):
    # 对应一些特殊的实体，如时间，该实体的kb_id为字面值，而不是 freebase mid
    if not is_freebase_mid(entity):
        return entity
    entity = entity[4:-1]
    if entity not in mid2name or len(mid2name[entity]) == 0:
        return "[unknown entity]"  # TODO：这里是否需要改变
    else:
        names = mid2name[entity]
        en_name = None
        for name in names:
            if name.endswith("@en"):
                en_name = name[:-3].strip("\"")
        if en_name is not None:
            return en_name
        elif "@" in names[0]:
            idx = names[0].find("@")
            return names[0][:idx].strip("\"")

def convert_relation(relation):
    assert relation.startswith("<fb:") and relation.endswith(">")
    return relation[4:-1]

def convert_path(path, mid2name):
    assert len(path) in (3, 5)
    if len(path) == 3:
        return (convert_entity(path[0], mid2name), convert_relation(path[1]), convert_entity(path[2], mid2name))
    elif len(path) == 5:
        return (convert_entity(path[0], mid2name), convert_relation(path[1]), convert_entity(path[2], mid2name),
                convert_relation(path[3]), convert_entity(path[4], mid2name))
        

def generate_candidates(entities, answers, subgraph, mid2name):
    if len(entities) == 0:
        return []
    candidates = []
    topic_entity_kb_id = entities[0]["kb_id"]
    answers_kb_id = set([answer["kb_id"] for answer in answers])
    subgraph_entities_kb_id = set([entity["kb_id"] for entity in entities])
    assert is_freebase_mid(topic_entity_kb_id)  # topic entity 都是 freebase mid 的形式
    assert topic_entity_kb_id in subgraph_entities_kb_id  # topic entity 都在子图中
    
    # 先把里面一层冗余的 kb_id: xxx, text: xxx 去掉，只考虑kb_id （因为kb_id和text似乎都是一样的）
    tuples = []
    for sub, rel, obj in subgraph["tuples"]:
        tuples.append((sub["kb_id"], rel["rel_id"], obj["kb_id"]))
    
    # 将所有三元组用头实体 or 尾实体进行索引，如果三元组是单向的话，应该只考虑头实体（即出发节点）
    indexed_tuples = collections.defaultdict(list)
    for _tuple in tuples:
        indexed_tuples[_tuple[0]].append(_tuple)
        indexed_tuples[_tuple[2]].append(reversed_tuple(_tuple))  # TODO: 有必要保留双向的吗？去掉反向的试试看？
    
    # 从 topic entity 开始，走一跳或者两跳，找到所有的不同的路径
    one_hop_paths = indexed_tuples[topic_entity_kb_id]
    two_hop_paths = []
    for one_hop_tuple in one_hop_paths:
        for two_hop_tuple in indexed_tuples[one_hop_tuple[2]]:
            two_hop_paths.append((one_hop_tuple[0], one_hop_tuple[1], one_hop_tuple[2], two_hop_tuple[1], two_hop_tuple[2]))
    candidate_indexed_paths = collections.defaultdict(list)
    for path in one_hop_paths + two_hop_paths:
        candidate_indexed_paths[path[-1]].append(path)
    
    # 最终对路径进行处理，候选实体是否为答案、候选实体是否为中心实体、是否需要downsampling、将mid替换为实体name等等
    for subgraph_entity in subgraph["entities"]:
        entity_kb_id = subgraph_entity["kb_id"]
        raw_paths = candidate_indexed_paths[entity_kb_id]
        random.shuffle(raw_paths)
        raw_paths = raw_paths[:10] # TODO: 有必要下采样吗？
        converted_paths = []
        for raw_path in raw_paths:
            converted_paths.append(convert_path(raw_path, mid2name))
        candidates.append({
            "candidate_id": entity_kb_id,
            "is_answer": int(entity_kb_id in answers_kb_id),
            "is_self": int(entity_kb_id == topic_entity_kb_id),
            "converted_paths": converted_paths,
            "raw_paths": raw_paths
        })
        candidates = sorted(candidates, key=lambda item: item["is_answer"], reverse=True)
    return candidates
        

def main():
    random.seed(1)
    
    subgraph_file = "../GraftNet/preprocessing/webqsp_subgraphs.json"
    mid2name_file = "../freebase/mid2name.json"
    output_train_file = "./train.json"
    output_dev_file = "./dev.json"
    output_test_file = "./test.json"
    
    print("Loading mid2name file...")
    mid2name = json.load(open(mid2name_file))
    print("Loading subgraph file...")
    question_subgraphs = [json.loads(line.strip()) for line in open(subgraph_file).readlines()]
    
    train_samples = {}
    test_samples = {}
    for question_subgraph in tqdm.tqdm(question_subgraphs):
        question, entities, answers, q_id, subgraph = (question_subgraph[key] for key in ("question", "entities", "answers", "id", "subgraph"))
        results = {
            "question": question,
            "is_subgraph_empty": len(entities) == 0,
            "candidiates": generate_candidates(entities, answers, subgraph, mid2name)
        }
        if q_id.startswith("WebQTrn"):
            train_samples[q_id] = results
        elif q_id.startswith("WebQTest"):
            test_samples[q_id] = results

    all_train_ids = list(train_samples.keys())
    random.shuffle(all_train_ids)
    # 250 samples for dev dataset
    dev_samples_after_split = {q_id: train_samples[q_id] for q_id in all_train_ids[:250]}
    train_samples_after_split = {q_id: train_samples[q_id] for q_id in all_train_ids[250:]}
    
    json.dump(train_samples_after_split, open(output_train_file, "w"))
    json.dump(dev_samples_after_split, open(output_dev_file, "w"))
    json.dump(test_samples, open(output_test_file, "w"))
    
    
if __name__ == "__main__":
    main()