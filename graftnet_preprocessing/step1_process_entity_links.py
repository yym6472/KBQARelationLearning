"""Script to process STAGG links into format for retrieving subgraphs.

Filters any non-top entity which has overlapping span with a higher ranked
entity and any non-top entity which has score lower than MIN_SCORE.
"""

questions_file = "scratch/webqsp_processed.json"
link_files = ["train_links_raw", "test_links_raw"]
entity_file = "freebase_2hops/all_entities"
output = "scratch/stagg_linked_questions.txt"

MIN_SCORE = 10.

import json
from tqdm import tqdm

def _convert_freebase_id(x):
    return "<fb:" + x[1:].replace("/", ".") + ">"

def _overlap(span, spans):
    for sp in spans:
        if max(sp[0], span[0]) < min(sp[1], span[1]):
            return True
    return False

def _filter_links(links):
    f_links = [links[0][:2]]
    spans_covered = [links[0][2:]]
    for item in links[1:]:
        if float(item[1]) < MIN_SCORE: continue
        if _overlap(item[2:], spans_covered): continue
        f_links.append(item[:2])
        spans_covered.append(item[2:])
    return f_links

# 所有的entity，从freebase_2hops/all_entities中读取
entities = set()
with open(entity_file) as f:
    for line in tqdm(f):
        entities.add(line.strip())

# 所有的问题id，从step0中处理的文件中读取
question_ids = set()
with open(questions_file) as f:
    all_questions = json.load(f)
    question_ids.update([q["QuestionId"] for q in all_questions])

# 处理链接文件，过滤掉不存在的实体，以及过滤掉不存在的问题id
entity_map = {qId: [] for qId in question_ids}
f_out = open(output, "w")
for fil in link_files:
    with open(fil) as f:
        for line in f:
            (qId, surface, start,
             length, fId, fsurface, score) = line.strip().split("\t")
            if _convert_freebase_id(fId) not in entities: continue
            if qId in entity_map:
                entity_map[qId].append([fId, score, start, start + length])

# 过滤掉分数过低、和之前实体有覆盖的实体，然后把结果按照每个问题一行的格式，输出到文件中
with open(output, "w") as f_out:
    for (qId, links) in entity_map.items():
        if not links:
            f_out.write("%s\t\n" % qId)
            continue
        link_str = ",".join(
            "%s=%s" % (_convert_freebase_id(fId), score)
            for fId, score in _filter_links(links))
        f_out.write("%s\t%s\n" % (qId, link_str))
