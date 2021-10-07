import json
import random


def main():
    """
    Generate fewrel for matching data.
    """
    
    train_file = "fewrel_train.json"
    dev_file = "fewrel_val.json"
    
    train_samples = json.load(open(train_file))
    dev_samples = json.load(open(dev_file))
    all_samples = {}
    all_samples.update(train_samples)
    all_samples.update(dev_samples)
    assert len(all_samples) == 80
    
    all_relations = set(all_samples.keys())
    
    generated_dataset = []
    for p_id, data_list in all_samples.items():
        assert len(data_list) == 700
        relations_exclude_current = list(all_relations - set([p_id]))
        assert len(relations_exclude_current) == 79
        
        for sample in data_list:
            exclude_self = list(data_list)
            exclude_self.remove(sample)
            positive_sample = random.sample(exclude_self, 1)[0]
            assert positive_sample is not sample
            generated_dataset.append({
                "text1": " ".join(sample["tokens"]),
                "text2": " ".join(positive_sample["tokens"]),
                "relation1": p_id,
                "relation2": p_id,
                "label": 1
            })
            
            for _ in range(9):
                neg_relation = random.sample(relations_exclude_current, 1)[0]
                neg_sample = random.sample(all_samples[neg_relation], 1)[0]
                assert neg_relation != p_id
                generated_dataset.append({
                    "text1": " ".join(sample["tokens"]),
                    "text2": " ".join(neg_sample["tokens"]),
                    "relation1": p_id,
                    "relation2": neg_relation,
                    "label": 0
                })
    random.shuffle(generated_dataset)
    generated_dev_dataset = generated_dataset[:10000]
    invalid_texts = set()
    for sample in generated_dev_dataset:
        invalid_texts.add(sample["text1"])
        invalid_texts.add(sample["text2"])
    generated_train_dataset = []
    for sample in generated_dataset[10000:]:
        if sample["text1"] in invalid_texts or sample["text2"] in invalid_texts:
            continue
        generated_train_dataset.append(sample)
    
    print(f"Number of train samples: {len(generated_train_dataset)}")
    json.dump(generated_train_dataset, open("../fewrel_train.json", "w"))
    print(f"Number of dev&test samples: {len(generated_dev_dataset)}")
    json.dump(generated_dev_dataset, open("../fewrel_dev_and_test.json", "w"))


def merge_with_webred():
    fewrel_train = "../fewrel_train.json"
    fewrel_test = "../fewrel_dev_and_test.json"
    webred_train = "../../webred_matching/webred_21_for_matching.json"
    webred_test = "../../webred_matching/webred_5_for_matching.json"
    
    def merge_single(file_a, file_b, output_file):
        data_a = json.load(open(file_a))
        data_b = json.load(open(file_b))
        data = data_a + data_b
        random.shuffle(data)
        print(f"Number of samples: {len(data)}")
        json.dump(data, open(output_file, "w"))
    
    merge_single(fewrel_train, webred_train, "../webred_fewrel_train.json")
    merge_single(fewrel_test, webred_test, "../webred_fewrel_dev_and_test.json")
    

if __name__ == "__main__":
#    main()
    merge_with_webred()