"""
该脚本使用WebQSP对应的召回子图（设置子图大小为500的版本，文件为：
/home/hadoop-aipnlp/cephfs/data/yanyuanmeng/kbqa/reproduce/data/GraftNet/preprocessing/webqsp_subgraphs.json）
和freebase对应的mid2name映射（文件：/home/hadoop-aipnlp/cephfs/data/yanyuanmeng/kbqa/reproduce/data/freebase/mid2name.json）
处理一个entity2name、relation2name的惟一映射矩阵，entity和relation均为所有子图中出现的实体/关系。
"""

import json


def is_freebase_mid(kb_id):
    return kb_id.startswith("<fb:") and kb_id.endswith(">")
    
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
    return relation[4:-1].replace(".", " ").replace("_", " ")

def write_to_file(mapping, file):
    with open(file, "w") as f:
        keys = list(mapping.keys())
        keys = sorted(keys)
        for key in keys:
            value = mapping[key]
            assert "\t" not in key and "\n" not in key
            assert "\t" not in value and "\n" not in value
            f.write(f"{key}\t{value}\n")

def main():
    subgraph_file = "/home/hadoop-aipnlp/cephfs/data/yanyuanmeng/kbqa/reproduce/data/GraftNet/preprocessing/webqsp_subgraphs.json"
    mid2name_file = "/home/hadoop-aipnlp/cephfs/data/yanyuanmeng/kbqa/reproduce/data/freebase/mid2name.json"
    subgraphs = [json.loads(line.strip()) for line in open(subgraph_file)]
    mid2name_mapping = json.load(open(mid2name_file))
    all_entities = set()
    all_relations = set()
    for subgraph in subgraphs:
        for e1, r, e2 in subgraph["subgraph"]["tuples"]:
            all_entities.add(e1["kb_id"])
            all_relations.add(r["rel_id"])
            all_entities.add(e2["kb_id"])
    
    mid2name = {}
    for entity in all_entities:
        mid2name[entity] = convert_entity(entity, mid2name_mapping)
    
    rid2name = {}
    for relation in all_relations:
        rid2name[relation] = convert_relation(relation)
    
    write_to_file(mid2name, "./entity2text.txt")
    write_to_file(rid2name, "./relation2text.txt")
    
    
if __name__ == "__main__":
    main()