import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoModel, AutoConfig
from torch.nn import CosineSimilarity
# from sentence_transformers.util import cos_sim


class HybridEnsemble(nn.Module):
    def __init__(self, ce_roberta_model_name, st_model_name, gist_model_name):
        super(HybridEnsemble, self).__init__()
        self.sts_config = AutoConfig.from_pretrained(ce_roberta_model_name)

        self.multilingual_st = AutoModel.from_pretrained(st_model_name)
        self.gist_bert = AutoModel.from_pretrained(gist_model_name)
        self.CE_roberta = AutoModelForSequenceClassification.from_pretrained(
            ce_roberta_model_name, config=self.sts_config
        )
        self.linear_last = nn.Linear(3, 1)

    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(
        self,
        roberta_features,
        st_features_0,
        st_features_1,
        gist_features_0,
        gist_features_1,
    ):
        sts_predictions = self.CE_roberta(**roberta_features, return_dict=True)
        # activation_fn = nn.Sigmoid()
        activation_fn = nn.Identity()
        sts_similarity = activation_fn(sts_predictions.logits)

        st_output_0 = self.multilingual_st(**st_features_0, return_dict=True)
        st_embeddings_0 = self.average_pool(
            st_output_0.last_hidden_state, st_features_0["attention_mask"]
        )
        st_output_1 = self.multilingual_st(**st_features_1, return_dict=True)
        st_embeddings_1 = self.average_pool(
            st_output_1.last_hidden_state, st_features_1["attention_mask"]
        )

        gist_output_0 = self.gist_bert(**gist_features_0, return_dict=True)
        gist_embeddings_0 = self.average_pool(
            gist_output_0.last_hidden_state, gist_features_0["attention_mask"]
        )
        gist_output_1 = self.gist_bert(**gist_features_1, return_dict=True)
        gist_embeddings_1 = self.average_pool(
            gist_output_1.last_hidden_state, gist_features_1["attention_mask"]
        )

        cos_sim = CosineSimilarity()
        st_similarity = cos_sim(st_embeddings_0, st_embeddings_1)
        gist_similarity = cos_sim(gist_embeddings_0, gist_embeddings_1)

        prediction = self.linear_last(
            F.relu(torch.cat((sts_similarity, st_similarity, gist_similarity), dim=1))
        )

        return prediction
