import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as tv
import torchvision.datasets as datasets

transform = tv.Compose([tv.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)

batch_size = 32
shuffle = True
num_workers = 0


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class MyModel(torch.nn.Module):
    def __init__(self, num_labels):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(self.fc2.out_features, num_labels, bias=False)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten the input image
        x = torch.relu(self.fc1(x))
        x_fc2 = torch.relu(self.fc2(x))
        x_fc2_norm = F.normalize(x_fc2)

        # normalize the rows of fc3 weights
        norm = self.fc3.weight.norm(p=2, dim=1, keepdim=True)
        self.fc3.weight = nn.Parameter(self.fc3.weight.div(norm))

        x = self.fc3(x_fc2_norm)
        return x

# model = MyModel(num_labels=10)
# input_tensor = torch.randn(1, 1, 28, 28)  # create a random input tensor
# output_tensor_fc3, output_tensor_fc2 = model(input_tensor)
# print(output_tensor_fc3.shape)
# print(output_tensor_fc2.shape)
# print(model.fc3.weight)
# print(model.fc3.weight.shape)

def custom_loss(logits, labels, s=10.0, m=0.5):
    theta_y = torch.arccos(logits[torch.arange(logits.size(0)), labels])
    theta_y_m = theta_y + m
    cos_theta_y_m = torch.cos(theta_y_m)
    cos_theta_j = logits
    exp_s_cos_theta = torch.exp(s * cos_theta_j)
    sum_exp_s_cos_theta = exp_s_cos_theta.sum(dim=1)
    loss = -torch.log(torch.exp(s * cos_theta_y_m) / (torch.exp(s * cos_theta_y_m) + sum_exp_s_cos_theta))
    return loss.mean()

logger = TensorBoardLogger('logs/')

def accuracy(logits, labels):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc

# create a new directory for the logs
log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(log_dir, exist_ok=True)

class MyLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, lr=5e-5, s=10.0, m=0.5):
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
        logits = self.model(input_ids_padded)
        loss = custom_loss(logits, labels, self.s, self.m)
        ###### 
        acc = accuracy(logits, labels)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        self.writer.add_scalar('Train/Loss', loss, self.global_step)
        self.writer.add_scalar('Train/Accuracy', acc, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids_padded = batch[0]
        labels = batch[1]
        logits = self.model(input_ids_padded)
        loss = custom_loss(logits, labels, self.s, self.m)
        ###### 
        acc = accuracy(logits, labels)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        self.writer.add_scalar('Val/Loss', loss, self.global_step)
        self.writer.add_scalar('Val/Accuracy', acc, self.global_step)

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.lr)     
        return optimizer
    
model = MyModel(num_labels=10)
lightning_module = MyLightningModule(model, num_classes=10)

# print(list(lightning_module.parameters()))

trainer = pl.Trainer(max_epochs=3, logger=logger, log_every_n_steps=5)
trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
