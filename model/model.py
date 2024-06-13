import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class ReviewModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(ReviewModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.food_rating_head = nn.Linear(768, 5)  # 768 is the output size of BERT
        self.delivery_rating_head = nn.Linear(768, 5)
        self.approval_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token's output

        food_logits = self.food_rating_head(pooled_output)

        delivery_logits = self.delivery_rating_head(pooled_output)

        approval_logit=self.approval_head(pooled_output)

        return food_logits, delivery_logits, approval_logit


    def from_logits(self,food_logits,delivery_logits,approval_logit):
        food_probs = F.softmax(food_logits, dim=-1)
        delivery_probs = F.softmax(delivery_logits, dim=-1)
        approval_prob = torch.sigmoid(approval_logit)
        return food_probs,delivery_probs,approval_prob

