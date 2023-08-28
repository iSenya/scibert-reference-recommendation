import os, json, langdetect
from torch.utils.data import Dataset
import torch.nn.functional as F
from preprocessing import strip_links, strip_all_entities
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer_class, pretrained_weights = (AutoTokenizer, 'allenai/scibert_scivocab_cased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

class MyDataset(Dataset):
    def __init__(self, input_ids_by_label):
        self.input_ids_by_label = input_ids_by_label
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.input_ids_by_label.keys()))
        self.key_value_pairs = []
        self.num_labels = len(input_ids_by_label)
        for label, input_ids_list in self.input_ids_by_label.items():
            label_int = self.label_encoder.transform([label])[0]
            for input_ids in input_ids_list:
                self.key_value_pairs.append((input_ids[0], label_int))

    def __len__(self):
        return len(self.key_value_pairs)

    def __getitem__(self, idx):
        input_ids, label  = self.key_value_pairs[idx]
        return input_ids, label

# Define directories
train_dir = "/Users/senyaisavnina/Downloads/thesis/scibert-reference-recommendation/dataset/train/"
test_dir = "/Users/senyaisavnina/Downloads/thesis/scibert-reference-recommendation/dataset/test/"

# Initialize the dictionary to store the abstracts by label
abstracts_by_label = {}

def process_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        label = data["paperId"]
        abstract = data["abstract"]
        if abstract is not None and isinstance(abstract, str):
            try:
                lang = langdetect.detect(abstract)
                if lang == 'en':
                    yield label, strip_all_entities(strip_links(str(abstract)))
                    for referenced_item in data["references"]:
                        ref_abstract = referenced_item.get("abstract")
                        if ref_abstract and isinstance(ref_abstract, str):
                            try:
                                lang = langdetect.detect(ref_abstract)
                                if lang == 'en':
                                    yield label, strip_all_entities(strip_links(str(ref_abstract)))
                            except langdetect.lang_detect_exception.LangDetectException:
                                pass
            except langdetect.lang_detect_exception.LangDetectException:
                pass

def preprocess_data(directory):
    input_ids_by_label = {}
    for filename in os.listdir(directory)[:50]:
        if filename.endswith(".json"):
            for label, abstract in process_file(os.path.join(directory, filename)):
                encoded_dict = tokenizer.encode_plus(
                    abstract,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    pad_to_max_length=True,
                    truncation=True,
                    return_attention_mask=False,
                    return_tensors='pt'
                )
                input_ids = encoded_dict['input_ids']
                if label not in input_ids_by_label:
                    input_ids_by_label[label] = []
                input_ids_by_label[label].append(input_ids)
    return input_ids_by_label

# Preprocess train and test datasets
train_dataset = MyDataset(preprocess_data(train_dir))
test_dataset = MyDataset(preprocess_data(test_dir))
