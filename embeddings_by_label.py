import json, os, re, string, random, langdetect
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from preprocessing import strip_links, strip_all_entities

tokenizer_class, pretrained_weights = (AutoTokenizer, 'allenai/scibert_scivocab_cased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

file_paths = [
    "/Users/senyaisavnina/Downloads/extracted_data_w_citations.json",
    "/Users/senyaisavnina/Downloads/extracted_data_w_citations_1.json",
    "/Users/senyaisavnina/Downloads/extracted_data_w_citations_2.json",
]

full_dataset = []

for path in file_paths:
    with open(path, "r") as f:
        dataset = json.load(f)
        full_dataset.extend(dataset)

# print(len(full_dataset))

# Define the size of the random sample
sample_size = 500

# Create a random sample of the full dataset
random_sample = random.sample(full_dataset, sample_size)

# Initialize the dictionary to store the abstracts by label
abstracts_by_label = {}

# Iterate over each item in the random sample
for item in random_sample:
    label = item["paperId"]
    abstract = item["abstract"]
    if abstract and isinstance(abstract, str):
        try:
            # Detect the language of the abstract
            lang = langdetect.detect(abstract)
            # Check if the abstract is in English
            if lang == 'en':
                if label not in abstracts_by_label:
                    abstracts_by_label[label] = []
                abstracts_by_label[label].append(strip_all_entities(strip_links(str(abstract))))
                for referenced_item in item["referenced_abstracts"]:
                    if referenced_item and isinstance(referenced_item, str):
                        try:
                            # Detect the language of the referenced abstract
                            lang = langdetect.detect(referenced_item)
                            # Check if the referenced abstract is in English
                            if lang == 'en':
                                abstracts_by_label[label].append(strip_all_entities(strip_links(str(referenced_item))))
                        except langdetect.lang_detect_exception.LangDetectException:
                            pass
        except langdetect.lang_detect_exception.LangDetectException:
            pass

# Print the size of the resulting dictionary
print(len(abstracts_by_label))

num_labels = len(abstracts_by_label)

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

dataset_size = len(dataset)

print(dataset_size)

# spitting dataset into train and test

train_size = int(dataset_size * 0.8) # 80% of data for training
test_size = dataset_size - train_size # 20% of data for testing

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 32
shuffle = True
num_workers = 0

# print(len(dataset))
# print(len(dataset[0]))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class MyModel(torch.nn.Module):
    def __init__(self, num_labels, bert_model):
        super(MyModel, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(bert_model.config.hidden_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1.out_features, 256)
        self.fc3 = nn.Linear(self.fc2.out_features, num_labels, bias=False)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output, _ = torch.max(last_hidden_state, dim=1)
        hidden1 = self.fc1(pooled_output)
        hidden1 = self.relu(hidden1)
        hidden2 = self.fc2(hidden1)
        hidden2 = F.normalize(self.relu(hidden2))

        # normalize the rows of fc3 weights
        norm = self.fc3.weight.norm(p=2, dim=1, keepdim=True)
        self.fc3.weight = nn.Parameter(self.fc3.weight.div(norm))

        logits = self.fc3(hidden2)

        return logits, pooled_output, last_hidden_state

model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')    

batch = next(iter(train_dataloader))
# print("batch: ", batch)

input_ids = batch[0]
# print("input_ids: ", input_ids)

labels = batch[1]

print("labels: ", labels)
logits, _, _ = model2(input_ids)
print(logits.shape)

def custom_loss(logits, labels, s=10.0, m=0.5):
    theta_y = torch.arccos(logits[torch.arange(logits.size(0)), labels])
    theta_y_m = theta_y + m
    cos_theta_y_m = torch.cos(theta_y_m)
    cos_theta_j = logits
    exp_s_cos_theta = torch.exp(s * cos_theta_j)
    sum_exp_s_cos_theta = exp_s_cos_theta.sum(dim=1)
    loss = -torch.log(torch.exp(s * cos_theta_y_m) / 
                      (torch.exp(s * cos_theta_y_m) + 
                       sum_exp_s_cos_theta))
    return loss.mean()

def accuracy(logits, labels, top_k=10):
    """
    Computes the top-k accuracy of the predictions with respect to the true labels.
    
    Args:
        logits (torch.Tensor): A tensor of shape [batch_size, num_classes] containing the predicted logits.
        labels (torch.Tensor): A tensor of shape [batch_size] containing the true labels.
        top_k (int): The number of top predictions to consider (default: 10).
        
    Returns:
        float: The top-k accuracy of the predictions.
    """
    _, preds = logits.topk(top_k, dim=1)
    correct = (preds == labels.unsqueeze(1)).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc

logger = TensorBoardLogger('logs/')

# create a new directory for the logs
log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)

class MyLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, lr=3e-4, s=10.0, m=0.5):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.s = s
        self.m = m

        self.writer = SummaryWriter(log_dir=log_dir)

    def training_step(self, batch):
        input_ids_padded = batch[0]
        labels = batch[1]
        logits, _, _ = self.model(input_ids_padded)
        loss = custom_loss(logits, labels, self.s, self.m)
        acc = accuracy(logits, labels)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        self.writer.add_scalar('Train/Loss', loss, self.global_step)
        self.writer.add_scalar('Train/Accuracy', acc, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids_padded = batch[0]
        labels = batch[1]
        logits, _, _ = self.model(input_ids_padded)
        loss = custom_loss(logits, labels, self.s, self.m)
        ###### 
        acc = accuracy(logits, labels)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        self.writer.add_scalar('Val/Loss', loss, self.global_step)
        self.writer.add_scalar('Val/Accuracy', acc, self.global_step)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
model2 = MyModel(num_labels, model)
lightning_module = MyLightningModule(model2, num_labels)

trainer = pl.Trainer(max_epochs=20, logger=logger, log_every_n_steps=20)
trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)




# with open("embeddings_by_label_first10.json", "w") as f:
#   json.dump(embeddings_by_label, f)