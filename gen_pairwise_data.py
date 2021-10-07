# encoding=utf8
import json
import random
random.seed(40)

base_dir = "/home/hadoop-aipnlp/cephfs/data/yanyuanmeng/kbqa/reproduce/data"

def main():
#     for_webred_pretrain(f"{base_dir}/webred_matching/webred_5_for_matching.json", f"{base_dir}/webred_matching/webred_5_for_matching_pairwise.json")
#     for_webred_pretrain(f"{base_dir}/webred_matching/webred_21_for_matching.json", f"{base_dir}/webred_matching/webred_21_for_matching_pairwise.json")
    for_webqsp_kbqa(f"{base_dir}/webqsp_name_expanded/dev.json", f"{base_dir}/webqsp_name_expanded_pairwise/dev.json")
    for_webqsp_kbqa(f"{base_dir}/webqsp_name_expanded/train.json", f"{base_dir}/webqsp_name_expanded_pairwise/train.json")
    for_webqsp_kbqa(f"{base_dir}/webqsp_name_expanded/test.json", f"{base_dir}/webqsp_name_expanded_pairwise/test.json")
    
def for_webqsp_kbqa(in_file, out_file):
    samples = json.load(open(in_file, "r"))
    cnt = 0
    new_samples = {}
    for q_id, sample in samples.items():
        if sample["is_subgraph_empty"]:
            continue
        question = sample["question"]
        pos_samples = [x for x in sample["candidiates"] if x["is_answer"]]
        if len(pos_samples)<1: continue
        new_candidates = []
        for candidate in sample["candidiates"]:
            if candidate["is_answer"] or len(candidate["converted_paths"])<1:
                continue
            random.shuffle(pos_samples)
            candidate["converted_paths_pos"] = pos_samples[0]["converted_paths"]
            candidate["is_self_pos"] = pos_samples[0]["is_self"]
            cnt += 1
            new_candidates.append(candidate)
        sample["candidiates"] = new_candidates
        new_samples[q_id] = sample
    with open(out_file, "w") as fout:
        json.dump(new_samples, fout, ensure_ascii=False)
    print(len(samples), cnt)
    
def for_webred_pretrain(in_file, out_file):
    dat = json.load(open(in_file))
    print(len(dat))
    pos_dict = {}
    for ln in dat:
        if ln["label"]==1:
            pos_dict[ln["text1"]] = ln["text2"]
    final_out = []
    for ln in dat:
        if ln["label"]==0 and ln["text1"] in pos_dict:
            final_out.append({"text":ln["text1"], "pos":pos_dict[ln["text1"]], "neg":ln["text2"]})
    print(len(final_out))
    with open(out_file, "w") as fout:
        json.dump(final_out, fout, ensure_ascii=False)
    
if __name__=="__main__":
    main()