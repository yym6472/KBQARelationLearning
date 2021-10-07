import gzip
import tqdm
import json
import collections

def read_entities(filename):
    all_entities = set()
    with open(filename, "r") as f:
        for line in tqdm.tqdm(f):  # total 22680702
            line = line.strip()
            if not line.startswith("<fb:"):
                continue
            if not line.endswith(">"):
                continue
            all_entities.add(line[4:-1].strip())
    return all_entities
            
def main():
    freebase_dump_file = "./freebase-rdf-latest.gz"
    all_entities_file = "../GraftNet/preprocessing/freebase_2hops/all_entities"
    all_entities = read_entities(all_entities_file)
    
    f_in = gzip.open(freebase_dump_file, "rb")
    data = collections.defaultdict(list)
    
    source_idx = 0
    added_idx = 0
    for line in tqdm.tqdm(f_in):  # total 3130753066
        source_idx += 1
        fields = line.decode("utf-8").strip().split("\t")[:3]
        if len(fields) != 3:
            continue
        if fields[1] != "<http://rdf.freebase.com/ns/type.object.name>":
            continue
        if not fields[0].startswith("<http://rdf.freebase.com/ns/"):
            continue
        if not fields[0].endswith(">"):
            continue
        if fields[0][28:-1] not in all_entities:
            continue
        if fields[2].strip() in data[fields[0][28:-1]]:
            continue
        data[fields[0][28:-1]].append(fields[2].strip())
        added_idx += 1
        if added_idx % 100000 == 0:
            json.dump({
                "source_idx": source_idx,
                "added_idx": added_idx
            }, open("./metadata.json", "w"))
    json.dump(data, open("./mid2name.json", "w"))
    f_in.close()
    

if __name__ == "__main__":
    main()