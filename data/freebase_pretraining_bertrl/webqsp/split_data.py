import random
random.seed(1)


def write_to_file(lines, filename):
    with open(filename, "w") as f_out:
        for line in lines:
            f_out.write(line)

def main():
    input_file = "./all_data.tsv"
    output_train = "./train.tsv"
    output_dev = "./dev.tsv"
    output_test = "./test.tsv"
    num_dev = 100000
    num_test = 100000
    
    with open(input_file) as f:
        lines = f.readlines()
    random.shuffle(lines)
    
    dev_lines = lines[0 : num_dev]
    test_lines = lines[num_dev : num_dev + num_test]
    train_lines = lines[num_dev + num_test : len(lines)]
    
    write_to_file(train_lines, output_train)
    write_to_file(dev_lines, output_dev)
    write_to_file(test_lines, output_test)
    

if __name__ == "__main__":
    main()