import torch
import torch.nn as nn

class ModelForEval(nn.Module):
    def __init__(self, bert_model):
        super(ModelForEval, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(bert_model.config.hidden_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, bert_model.config.hidden_size)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.max(last_hidden_state, dim=1)[0]  # Get the maximum values only
        hidden1 = self.fc1(pooled_output)
        hidden1 = self.relu(hidden1)
        hidden2 = self.fc2(hidden1)
        hidden2 = nn.functional.normalize(hidden2)
        embedding_vector = hidden2
        return embedding_vector
