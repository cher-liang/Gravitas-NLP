import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class QuestionAnswerPairSimilarityModel(nn.Module):
    # def __init__(self, lstm_dim, qnli_model_name, sts_model_name):
    def __init__(self, embedding_dim, hidden_dim, qnli_model_name, sts_model_name):
        super(QuestionAnswerPairSimilarityModel, self).__init__()

        self.qnli_bert = AutoModelForSequenceClassification.from_pretrained(
            qnli_model_name
        )
        self.sts_bert = AutoModelForSequenceClassification.from_pretrained(
            sts_model_name
        )

        # self.bilstm = nn.LSTM(
        #     self.qnli_bert.config.hidden_size, lstm_dim, bidirectional=True
        # )

        # self.linear = nn.Linear(2 * lstm_dim, self.qnli_bert.config.hidden_size)

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, embedding_dim)

        self.linear_last = nn.Linear(2, 1)

    def forward(
        self,
        question_reference_answer_features,
        question_student_answer_features,
        reference_answer_student_answer_features,
    ):
        qnli_predictions1 = self.qnli_bert.base_model(
            **question_reference_answer_features, return_dict=True
        )
        qnli_predictions2 = self.qnli_bert.base_model(
            **question_student_answer_features, return_dict=True
        )

        # embeddings1 = qnli_predictions1[0][:,0,:]
        # embeddings2 = qnli_predictions2[0][:,0,:]

        embeddings1 = qnli_predictions1[0]
        embeddings2 = qnli_predictions2[0]

        # # Pass the embeddings through the BiLSTM
        # embeddings1, _ = self.bilstm(embeddings1.unsqueeze(0))
        # embeddings2, _ = self.bilstm(embeddings2.unsqueeze(0))

        # # Pass the embeddings through the linear layer
        # embeddings1 = self.linear(embeddings1.squeeze(0))
        # embeddings2 = self.linear(embeddings2.squeeze(0))

        # Transform the embeddings
        embeddings1 = F.relu(self.linear1(embeddings1))
        embeddings1 = F.relu(self.linear2(embeddings1))
        embeddings1 = self.linear3(embeddings1)

        embeddings2 = F.relu(self.linear1(embeddings2))
        embeddings2 = F.relu(self.linear2(embeddings2))
        embeddings2 = self.linear3(embeddings2)

        embeddings1 = embeddings1.mean(dim=1)
        embeddings2 = embeddings2.mean(dim=1)

        # Calculate the cosine similarity
        question_answer_similarity = F.cosine_similarity(embeddings1, embeddings2)

        activation_fn = nn.Sigmoid()
        sts_predictions = self.sts_bert(
            **reference_answer_student_answer_features, return_dict=True
        )
        sts_similarity = activation_fn(sts_predictions.logits)

        prediction = self.linear_last(
            torch.stack([question_answer_similarity, sts_similarity.view(-1)], dim=1)
        )

        return prediction


# model = QuestionAnswerPairSimilarityModel(768, 256)
