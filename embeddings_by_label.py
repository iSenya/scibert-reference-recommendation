import json
from transformers import *
import re, string
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils import weight_norm

tokenizer_class, pretrained_weights = (AutoTokenizer, 'allenai/scibert_scivocab_cased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

with open("/Users/senyaisavnina/Downloads/extracted_data_w_citations.json", "r") as f:
        dataset = json.load(f)[:10]

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = " ".join(word.strip().lower() for word in re.split('#|_', word))
        if word:
            if word not in entity_prefixes:
                words.append(word)
    return ' '.join(words)



abstracts_by_label = {}
for item in dataset:
    label = item["paperId"]
    abstract = item["abstract"]
    if label not in abstracts_by_label:
        abstracts_by_label[label] = []
    abstracts_by_label[label].append(strip_all_entities(strip_links(str(abstract))))
    for referenced_item in item["referenced_abstracts"]:
      if referenced_item is not None:
        abstracts_by_label[label].append(strip_all_entities(strip_links(str(referenced_item))))
print(len(abstracts_by_label))



input_ids_by_label = {}
for label, referenced_abstracts in abstracts_by_label.items():
    input_ids_by_label[label] = []
    for abstract in referenced_abstracts:
        encoded_dict = tokenizer.encode_plus(
            abstract,                             # Abstract to encode.
            add_special_tokens=True,              # Add '[CLS]' and '[SEP]'
            max_length=512,
            padding='max_length',                       # Pad & truncate all sentences.
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=False,          # Construct attn. masks.
            return_tensors='pt'                   # Return pytorch tensors.
        )
        input_ids = encoded_dict['input_ids']
        input_ids_by_label[label].append(input_ids)

# print(pooled_output)

class MyDataset(Dataset):
    def __init__(self, input_ids_by_label):
        self.input_ids_by_label = input_ids_by_label

        # Convert string labels to integer labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.input_ids_by_label.keys()))

        # Create list of key-value pairs
        self.key_value_pairs = []
        for label, input_ids_list in self.input_ids_by_label.items():
            label_int = self.label_encoder.transform([label])[0]
            for input_ids in input_ids_list:
                self.key_value_pairs.append((input_ids[0], label_int))

    def __len__(self):
        return len(self.key_value_pairs)

    def __getitem__(self, idx):
        input_ids, label  = self.key_value_pairs[idx]

        return input_ids, label


dataset = MyDataset(input_ids_by_label)

batch_size = 32
shuffle = True
num_workers = 0

# print(len(dataset))
print(len(dataset[0]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class MyModel(torch.nn.Module):
    def __init__(self, num_labels, bert_model):
        super(MyModel, self).__init__()
        self.bert = bert_model
        self.fc = weight_norm(torch.nn.Linear(bert_model.config.hidden_size, num_labels, bias = False))

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output, _ = torch.max(last_hidden_state, dim=1)
        pooled_output = F.normalize(pooled_output)
        logits = self.fc(pooled_output)
        # normalized_logits = F.normalize(logits, p=2, dim=1)

        return logits, pooled_output, last_hidden_state

model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')    
model2  = MyModel(10, model)

batch = next(iter(dataloader))
print("batch: ", batch)
input_ids = batch[0]
print("input_ids: ", input_ids)
labels = batch[1]
print("labels: ", labels)
# break
logits, _, _ = model2(input_ids)
print(logits.shape)

def custom_loss(logits, labels, s=10.0, m=0.5):
    theta_y = torch.arccos(logits[torch.arange(logits.size(0)), labels])
    theta_y_m = theta_y + m
    cos_theta_y_m = torch.cos(theta_y_m)
    cos_theta_j = logits
    exp_s_cos_theta = torch.exp(s * cos_theta_j)
    sum_exp_s_cos_theta = exp_s_cos_theta.sum(dim=1)
    loss = -torch.log(torch.exp(s * cos_theta_y_m) / (torch.exp(s * cos_theta_y_m) + sum_exp_s_cos_theta))
    return loss.mean()

import pytorch_lightning as pl

class MyLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, lr=1e-5, s=10.0, m=0.5):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.s = s
        self.m = m

    def training_step(self, batch):
        input_ids_padded = batch[0]
        labels = batch[1]
        logits, _, _ = self.model(input_ids_padded)
        loss = custom_loss(logits, labels, self.s, self.m)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
model2 = MyModel(10, model)
lightning_module = MyLightningModule(model2, 10)

trainer = pl.Trainer(max_epochs=3)
trainer.fit(lightning_module, dataloader)



# with open("embeddings_by_label_first10.json", "w") as f:
#   json.dump(embeddings_by_label, f)
