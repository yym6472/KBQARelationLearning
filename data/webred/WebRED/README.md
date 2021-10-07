# WebRED - Web Relation Extraction Dataset

A dataset for extracting relationships from a variety of text found on the World Wide Web.
Text on the web has diverse surface forms including writing styles, complexity and grammar.
This dataset collects sentences from a variety of webpages and documents that represent
a variety of those categories.
In each sentence, there will be a subject and object entities tagged with subject
`SUBJ{...}` and object `OBJ{...}`, respectively.
The two entities are either related by a relation from a set of pre-defined ones
or has no relation.

More information about the dataset can be found in
[our paper](https://arxiv.org/abs/2102.09681).
If you use this dataset, make sure you cite this work as:

```
@misc{ormandi2021webred,
    title={WebRED: Effective Pretraining And Finetuning For Relation Extraction On The Web}, 
    author={Robert Ormandi and Mohammad Saleh and Erin Winter and Vinay Rao},
    year={2021},
    eprint={2102.09681},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2102.09681},
}
```

We compare our dataset against other publicly available relationship extraction
corpus in the table below. Notably, ours is the only dataset with text that is
found on the web, where the input sources and writing styles vary wildly.

We only present the human-annotated subsets from each dataset in the table
below. The numbers in this table might be different from the ones reported in
our paper as we were unable to release the full version of our dataset due to
legal concerns.

| Dataset                                              | No of relations     | No of examples |
|------------------------------------------------------|---------------------|----------------|
| [TACRED](https://nlp.stanford.edu/projects/tacred/)  | 42                  | 106,264        |
| [DocRED](https://github.com/thunlp/DocRED)           | 96                  | 63,427         |
| *WebRED  5*                                          | 523                 | 3,898          |
| *WebRED 2+1*                                         | 523                 | 107,819        |

Each example in `WebRED 5` was annotated by exactly `5` independent human
annotators. In `WebRED 2+1`, each example was annotated by `2` independent
annotators. If they disagreed, an additional annotator (`+1`) was assigned to
the example who also provided a disambiguating annotation.

In our paper, we used the `WebRED` data to fine-tune a model trained on a large
unsupervised dataset. The details of how data collection of the pre-training
data, the unsupervised model training, the supervised fine-tuning using
`WebRED 2+1` and evaluation on `WebRED 5` happend are described in the paper.
The paper also compares this model against others built on the datasets mentioned in the table 
above.

## Preparation
First, download the data onto your disk:

```bash
cd ~
git clone https://github.com/google-research-datasets/WebRED.git
cd WebRED/
```

## Using the data
The dataset is distributed in `Tensorflow.Example` format encoded as
[`TFRecord`](https://www.tensorflow.org/tutorials/load_data/tfrecord).

One can easily read the content of the dataset using
[Tensorflow's data API](https://www.tensorflow.org/api_docs/python/tf/data):

```python
import tensorflow as tf

path_to_webred = '...'          # Path to where the WebRED data was downloaded.

def read_examples(*dataset_paths):
  examples = []
  dataset = tf.data.TFRecordDataset(dataset_paths)
  for raw_sentence in dataset:
    sentence = tf.train.Example()
    sentence.ParseFromString(raw_sentence.numpy())
    examples.append(sentence)
  return examples

webred_sentences = read_examples(path_to_webred)
sentence = webred_sentences[0]  # As an instance of `tf.Example`.
```

Description of the features:

  * `num_pos_raters`: Number of unique human raters who thought that the
    sentence expresses the given relation.
  * `num_raters`: Number of unique human raters whou annotated the sentence fact pair
  * `relation_id`: The
    [WikiData relation ID](https://www.wikidata.org/wiki/Wikidata:Identifiers)
    of the fact.
  * `relation_name`: Human readable name of the relation of the fact.
  * `sentence`: The sentence with object and subject annotation in it.
  * `source name`: The name of the subject (source) entity.
  * `target_name`: The name of the object (target) entity.
  * `url`: Original url containing the annotated sentence.

The individual features of an example can be accessed e.g. by the following way:

```python
def get_feature(sentence, feature_name, idx=0):
  feature = sentence.features.feature[feature_name]
  return getattr(feature, feature.WhichOneof('kind')).value[idx]

annotated_sentence_text = get_feature(sentence, 'sentence').decode('utf-8')
relation_name = get_feature(sentence, 'relation_name').decode('utf-8')
empirical_probability_of_the_sentence_expresses_the_relation = (
    get_feature(sentence, 'num_pos_raters') /
    get_feature(sentence, 'num_raters'))
```
## License

This data is licensed by Google LLC under a [Creative Commons Attribution 4.0
International License](http://creativecommons.org/licenses/by/4.0/).
Users will be allowed to modify and repost it, and we encourage them to analyze
and publish research based on the data.

## Contact Us

If you have a technical question regarding the dataset, code or publication,
please create an issue in this repository. You may also reach us at
webred@google.com.
