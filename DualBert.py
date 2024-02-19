import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig


class DualBERT(nn.Module):
    def __init__(self, qnli_model_name, sts_model_name):
        super(DualBERT, self).__init__()
        self.qnli_config = AutoConfig.from_pretrained(qnli_model_name)
        self.sts_config = AutoConfig.from_pretrained(sts_model_name)

        self.qnli_bert = AutoModelForSequenceClassification.from_pretrained(
            qnli_model_name, config=self.qnli_config
        )
        self.sts_bert = AutoModelForSequenceClassification.from_pretrained(
            sts_model_name, config=self.sts_config
        )
        self.linear_last = nn.Linear(2, 1)

    def forward(
        self,
        question_student_answer_features,
        reference_answer_student_answer_features,
    ):
        sts_predictions = self.sts_bert(
            **reference_answer_student_answer_features, return_dict=True
        )
        qnli_predictions = self.qnli_bert(
            **question_student_answer_features, return_dict=True
        )

        # activation_fn = nn.Sigmoid()
        activation_fn = nn.Identity()

        qnli_similarity = activation_fn(qnli_predictions.logits)
        sts_similarity = activation_fn(sts_predictions.logits)

        prediction = self.linear_last(
            F.relu(torch.cat((qnli_similarity, sts_similarity), dim=1))
        )

        return prediction
