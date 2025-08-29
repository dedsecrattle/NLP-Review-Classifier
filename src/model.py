import torch.nn as nn
from transformers import DebertaForSequenceClassification


class DebertaClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-base", num_labels=2):
        super().__init__()
        self.deberta = DebertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def get_probabilities(self, logits):
        """Helper method to get probabilities when needed"""
        return self.softmax(logits)
