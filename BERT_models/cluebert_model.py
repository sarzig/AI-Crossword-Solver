import torch.nn as nn
from transformers import BertModel

class ClueBertRanker(nn.Module):
    def __init__(self, pretrained_model="sdeakin/cluebert-model"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.score = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.score(cls_output).squeeze(-1)
