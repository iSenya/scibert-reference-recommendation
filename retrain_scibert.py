import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from retrain_dataset import train_dataset, test_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)

print(train_dataset_size)
print(test_dataset_size)


batch_size = 8
shuffle = True
num_workers = 0

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class MyModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(MyModel, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output, _ = torch.max(last_hidden_state, dim=1)
        return F.normalize(pooled_output)


class MyLightningModule(pl.LightningModule):

    def __init__(self, model, num_labels, lr=3e-4, s=10.0, m=0.5):
        super().__init__()
        self.lr = lr
        self.s = s
        self.m = m

        self.model = model
        self.fc = nn.Linear(self.model.bert.config.hidden_size, num_labels, bias=False)
        self.num_labels = num_labels

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

    def custom_loss(self, logits, labels, s=10.0, m=0.5):
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
    
    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        # Use SciBERT to obtain embeddings
        embeddings = self.model(input_ids)
        
        # normalize the rows of fc weights
        norm = self.fc.weight.norm(p=2, dim=1, keepdim=True)
        self.fc.weight = nn.Parameter(self.fc.weight.div(norm))
        logits = self.fc(embeddings)
        loss = self.custom_loss(logits, labels, self.s, self.m)
        ###### 
        acc = self.accuracy(logits, labels)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        self.writer.add_scalar('Train/Loss', loss, self.global_step)
        self.writer.add_scalar('Train/Accuracy', acc, self.global_step)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch[0]
        labels = batch[1]
        embeddings = self.model(input_ids)
        
        norm = self.fc.weight.norm(p=2, dim=1, keepdim=True)
        self.fc.weight = nn.Parameter(self.fc.weight.div(norm))
        logits = self.fc(embeddings)
        loss = self.custom_loss(logits, labels)
        ###### 
        acc = self.accuracy(logits, labels)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        self.writer.add_scalar('Val/Loss', loss, self.global_step)
        self.writer.add_scalar('Val/Accuracy', acc, self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    
scibert = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')    
logger = TensorBoardLogger('logs/')
retrained_scibert = MyModel(scibert)
lightning_module = MyLightningModule(retrained_scibert, train_dataset.num_labels)

trainer = pl.Trainer(max_epochs=20, logger=logger, log_every_n_steps=20)
trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)