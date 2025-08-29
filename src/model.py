import torch.nn as nn
from transformers import DebertaForSequenceClassification


class DebertaClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=4):
        super().__init__()
        self.deberta = DebertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        logits = outputs.logits
        probs = self.softmax(logits)
        return logits, probs
