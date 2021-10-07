from typing import Optional

import torch

from transformers import BertModel


class BertForKBQA(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 pooling_strategy: str = "max_pooling",
                 dropout: Optional[float] = None):
        super(BertForKBQA, self).__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.linear = torch.nn.Linear(768, 1)
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        assert pooling_strategy in ("max_pooling", "cls_token", "mean_pooling")
        self.pooling_strategy = pooling_strategy
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        token_embeddings, pooled_output = bert_outputs[:2]
        if self.pooling_strategy == "max_pooling":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            pooled_output = torch.max(token_embeddings, 1)[0]
        elif self.pooling_strategy == "mean_pooling":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        if self.dropout is not None:
            pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).squeeze(dim=-1)
        return logits


class BertForPretraining(torch.nn.Module):
    def __init__(self,
                 pretraining_task: str,
                 model_name_or_path: str,
                 vocab_size: int = None,
                 pooling_strategy: str = "max_pooling",
                 dropout: Optional[float] = None):
        super(BertForPretraining, self).__init__()
        
        # matching: bert for sentence pair classification (input: input_ids, token_type_ids, attention_mask; output: logits)
        # masked_token_prediction: (input: input_ids, token_type_ids, attention_mask; output: the token level logits)
        # joint: use both matching and masked_token_prediction task to train the model
        self.pretraining_task = pretraining_task
        assert pretraining_task in ("matching", "masked_token_prediction", "joint", "matching_pw")        
        self.bert = BertModel.from_pretrained(model_name_or_path)
        
        if pretraining_task in ("matching", "joint", "matching_pw"):
            self.linear = torch.nn.Linear(768, 1)
            assert pooling_strategy in ("max_pooling", "cls_token", "mean_pooling")
            self.pooling_strategy = pooling_strategy
        if pretraining_task in ("masked_token_prediction", "joint"):
            self.token_linear = torch.nn.Linear(768, vocab_size)
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        token_embeddings, pooled_output = bert_outputs[:2]
        
        binary_logits, token_logits = None, None
        if self.pretraining_task in ("matching", "joint", "matching_pw"):
            if self.pooling_strategy == "max_pooling":
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                pooled_output = torch.max(token_embeddings, 1)[0]
            elif self.pooling_strategy == "mean_pooling":
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            if self.dropout is not None:
                pooled_output = self.dropout(pooled_output)
            binary_logits = self.linear(pooled_output).squeeze(dim=-1)
        if self.pretraining_task in ("masked_token_prediction", "joint"):
            if self.dropout is not None:
                token_embeddings = self.dropout(token_embeddings)
            token_logits = self.token_linear(token_embeddings)  # shape: (batch_size, sequence_length, vocab_size)
        return binary_logits, token_logits