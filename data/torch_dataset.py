import os

# import torch
from torch.utils.data import Dataset
# from transformers import AutoTokenizer

import pandas as pd

from basis.enums import TrainingType
from basis.logger import logger

def find_reference_answer(row):
    if row.reference_answer_best_match_id is not None:
        for reference_answer in row.reference_answers:
            if (
                row.reference_answer_best_match_id
                == reference_answer.reference_answer_id
            ):
                return reference_answer.text
    else:
        return row.reference_answers[0].text

class GravitasDataset(Dataset):
    def __init__(
        self,
        path,
        # tokenizer: AutoTokenizer,
        max_length: int = 512,
        train: TrainingType = TrainingType.TRAINING,
        transform=None,
    ):
        # self.tokenizer = tokenizer
        self.max_length = max_length

        if train == TrainingType.TRAINING:
            dataset_path = os.path.join(path, "train")
        elif train == TrainingType.TESTING_UNSEEN_ANSWERS:
            dataset_path = os.path.join(path, "test-unseen-answers")
        elif train == TrainingType.TESTING_UNSEEN_DOMAINS:
            dataset_path = os.path.join(path, "test-unseen-domains")
        elif train == TrainingType.TESTING_UNSEEN_QUESTIONS:
            dataset_path = os.path.join(path, "test-unseen-questions")
        else:
            logger.error("SEB dataset must have specificed data training type")

        if train == TrainingType.TESTING_UNSEEN_ANSWERS:
            self.questions = pd.read_pickle(os.path.join(path,"train", "questions.pkl"))
        else:
            self.questions = pd.read_pickle(os.path.join(dataset_path, "questions.pkl"))
        self.answers = pd.read_pickle(os.path.join(dataset_path, "answers.pkl"))

        self.data=pd.merge(self.answers, self.questions, on=["question_id"])
        self.data.rename(columns={"text_x":"answer_text","text_y":"question_text"},inplace=True)
        self.data["best_match_reference_answer"] = self.data.apply(find_reference_answer, axis=1)

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()

        # text = str(self.data.loc[idx, "text"])
        # label = self.data.loc[idx, "label"]

        # encoding = self.tokenizer(
        #     text,
        #     truncation=True,
        #     padding=True,
        #     max_length=self.max_length,
        #     return_tensors="pt",
        # )

        # return {
        #     "text": text,
        #     "input_ids": encoding["input_ids"].flatten(),
        #     "attention_mask": encoding["attention_mask"].flatten(),
        #     "label": torch.tensor(label, dtype=torch.long),
        # }
