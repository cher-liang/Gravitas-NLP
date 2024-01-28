import os
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import docx2txt
import numpy as np
import xlrd

from basis import config
from basis.enums import ScoreEncoding, TrainingType, ReferenceAnswerCategory
from core import utils

LABELS_2WAY = {"Wrong": 0, "Correct": 1}
LABELS_3WAY = {"Wrong": 0, "Partial": 1, "Correct": 2}
LABELS_5WAY = {
    "Wrong": 0,
    "Partially Wrong": 1,
    "Marginal": 3,
    "Partially Correct": 4,
    "Correct": 5,
}


@dataclass
class Answer:
    answer_id: str
    text: str
    scores: list[str if ScoreEncoding.TEXT else int]
    normalized_scores: float
    score_encoding: ScoreEncoding
    question_id: str
    reference_answer_best_match_id: str | None

    def __str__(self):
        string = "[ANSWER]\n"
        string += "ANSWER ID: {}\n".format(self.answer_id)
        string += "SCORES: {}\n".format(self.scores)
        string += "NORMALIZED SCORES: {}\n".format(self.normalized_scores)
        string += "SCORE ENCODING: {}\n".format(self.score_encoding.name)
        string += "QUESTION ID: {}\n".format(self.question_id)
        string += "{}{}\n".format(
            self.text[:100] if self.text else None,
            "..." if self.text and len(self.text) >= 100 else "",
        )
        return string


@dataclass
class ReferenceAnswer:
    reference_answer_id: int
    category: ReferenceAnswerCategory
    text: str

    def __str__(self):
        string = "REFERENCE ANSWER ID: {}\n".format(self.reference_answer_id)
        string += "\tCATEGORY: {}\n".format(self.category)
        string += "\t{}\n".format(self.text)

        return string


@dataclass
class Question:
    question_id: str
    domain: str
    text: str
    reference_answers: list[ReferenceAnswer]

    def __str__(self):
        string = "[QUESTION]\n"
        string += "DOMAIN: {}\n".format(self.domain)
        string += "QUESTION ID: {}\n".format(self.question_id)
        string += "{}\n".format(self.text)
        string += "REFERENCE ANSWERS:\n"
        for reference_answer in self.reference_answers:
            string += "\t{}\n".format(reference_answer)
        return string


class Dataset(object):
    def __init__(self, name: str, score_encoding, extra_data, path):
        self.name = name
        self.metadata = extra_data if extra_data is not None else dict()
        if "has_reference_answer" not in self.metadata:
            self.metadata["has_reference_answer"] = False
        if "has_specified_training_type" not in self.metadata:
            self.metadata["has_specified_training_type"] = False
        self.metadata["score_encoding"] = score_encoding
        self.questions: list[Question] = []
        self.answers: list[Answer] = []

        self.process(path)

    def __str__(self):
        string = "[DATASET]\n"
        string += "NAME: {}\n".format(self.name)
        string += "HAS_REFERENCE_ANSWER: {}\n".format(
            self.metadata.get("has_reference_answer")
        )
        string += "SCORE CLASSES: {}\n".format(self.metadata.get("score_classes"))
        string += "SCORE ENCODING: {}\n".format(self.metadata.get("score_encoding"))
        string += "QUESTION COUNT: {}\n".format(len(self.questions))
        string += "ANSWER COUNT: {}".format(len(self.answers))
        return string

    def process(self, path):
        raise NotImplementedError("write function not implemented")

    def get_question(self, question_id):
        return next((q for q in self.questions if q.question_id == question_id), None)


class SciEntsBank2013Task7(Dataset):
    def __init__(self, subset_name, container_folder_name, score_classes, root_path):
        self.subset_name = subset_name
        self.container_folder_name = container_folder_name
        super().__init__(
            name="SEB{}".format(subset_name),
            score_encoding=ScoreEncoding.TEXT,
            extra_data={"has_reference_answer": True, "score_classes": score_classes},
            path=root_path,
        )

    def scan(self, path):
        files_dict = {
            "train": [],
            "test-unseen-answers": [],
            "test-unseen-domains": [],
            "test-unseen-questions": [],
        }
        for sets in ["beetle", "sciEntsBank"]:
            for datasetType in files_dict:
                curr_dir = os.path.join(path, sets, datasetType)
                if self.subset_name == "5":
                    curr_dir = os.path.join(curr_dir, "Core")
                if os.path.exists(curr_dir):
                    for file in os.listdir(curr_dir):
                        file_path = os.path.join(curr_dir, file)
                        if (
                            os.path.isfile(file_path)
                            and os.path.splitext(file_path)[1] == ".xml"
                        ):
                            files_dict[datasetType].append(file_path)

        return files_dict

    def process(self, path):
        files_dict = self.scan(path)
        # print(files_dict)

        for datasetType, file_paths in files_dict.items():
            _questions = []
            _answers = []
            for file_path in file_paths:
                question_element = ElementTree.parse(file_path).getroot()
                question_text = question_element.find("questionText").text
                reference_answer_elements = question_element.find(
                    "referenceAnswers"
                ).findall("referenceAnswer")
                student_answer_elements = question_element.find(
                    "studentAnswers"
                ).findall("studentAnswer")

                # attributes
                for key, value in question_element.attrib.items():
                    if key == "id":
                        question_id = value
                    elif key == "module":
                        domain = value
                reference_answers = []
                for answer in reference_answer_elements:
                    for key, value in answer.attrib.items():
                        if key == "id":
                            reference_answer_id = value
                        elif key == "category":
                            reference_answer_category = value
                    reference_answer_text = answer.text

                    reference_answers.append(
                        ReferenceAnswer(
                            reference_answer_id,
                            reference_answer_category,
                            reference_answer_text,
                        )
                    )

                if self.get_question(question_id) is None:
                    _questions.append(
                        Question(
                            question_id,
                            domain,
                            question_text,
                            reference_answers,
                        )
                    )

                for answer in student_answer_elements:
                    answer_text = answer.text
                    scores = []
                    answer_match = None
                    for key, value in answer.attrib.items():
                        if key == "id":
                            answer_id = value
                        elif key == "accuracy":
                            scores.append(value)
                        elif key == "answerMatch":
                            answer_match = value

                    score_nums = [
                        self.metadata["score_classes"][score] for score in scores
                    ]
                    normalized_scores = (
                        sum(score_nums)
                        / len(score_nums)
                        / max(self.metadata["score_classes"].values())
                    )

                    _answers.append(
                        Answer(
                            answer_id,
                            answer_text,
                            scores,
                            normalized_scores,
                            ScoreEncoding.TEXT,
                            question_id,
                            answer_match,
                        )
                    )

            self.questions.extend(_questions)
            self.answers.extend(_answers)

            save_dir = os.path.join(path, "processed", datasetType)
            utils.create_directories(save_dir)
            utils.dump_dataset(_questions, _answers, save_dir)


# class USCIS(Dataset):
#     def __init__(self, root_path, include_100=False):
#         self.file_names = ["questions_answer_key", "studentanswers_grades_698"]
#         if include_100:
#             self.file_names.append("studentanswers_grades_100")
#         self.fixes_path = Path(config.ROOT_PATH) / "data/datasets/fixes/USCIS"
#         self.grade_corrections_path = self.fixes_path / "grade_corrections.csv"
#         super().__init__(
#             name="USCIS{}".format("_include_100" if include_100 else ""),
#             score_encoding=ScoreEncoding.NUMERIC,
#             extra_data={"has_reference_answer": True, "score_classes": [-1, 0, 1]},
#             path=root_path,
#         )
#
#     def parse_questions(self, rows):
#         self.questions = []
#         domain = "USCIS"
#
#         i = 0
#         for row in rows:
#             if i == 0:
#                 i += 1
#                 continue
#
#             question_id = row[0]
#             question_text = row[1]
#             answer_text = row[2:]
#
#             question = Question(
#                 question_id=question_id, domain=domain, text=question_text
#             )
#
#             for j in range(0, len(answer_text)):
#                 question.answers.append(
#                     Answer(
#                         answer_id="{}_ref_{}".format(question.id, j + 1),
#                         text=answer_text[j].lower(),
#                         scores=[1, 1, 1],
#                         is_reference=True,
#                     )
#                 )
#
#             self.questions.append(question)
#
#     def parse_student_answers(self, rows, corrections):
#         i = 0
#         for row in rows:
#             if i > 0:
#                 question = next((q for q in self.questions if q.id == row[1]), None)
#                 # fix = None
#
#                 # read fixes and corrrect
#                 fix = next(
#                     (
#                         f
#                         for f in corrections[1]
#                         if question.id == f[1] and row[0] == f[2]
#                     ),
#                     None,
#                 )
#
#                 if fix is not None:
#                     scores = list(
#                         map(int, fix[-1].replace("[", "").replace("]", "").split(","))
#                     )
#                 else:
#                     scores = [int(score) for score in row[3:]]
#
#                 answer_id = "{}_student_{}".format(
#                     question.id, len(question.answers) + i
#                 )
#                 answer_text = row[2]
#
#                 if utils.is_empty_string(answer_text):
#                     print(
#                         "Empty answer! question ID: {}, line: {}, answer_id: {}".format(
#                             question.id, i, answer_id
#                         )
#                     )
#                     continue
#
#                 # question.answers.append(Answer(answer_id=answer_id, text=answer_text.lower(), scores=[int(score) for score in row[3:]], is_reference=False, metadata={'student': row[0]}))
#                 question.answers.append(
#                     Answer(
#                         answer_id=answer_id,
#                         text=answer_text.lower(),
#                         scores=scores,
#                         is_reference=False,
#                         metadata={"student": row[0]},
#                     )
#                 )
#             i += 1
#
#     def read_grade_corrections(self):
#         rows = utils.read_csv_file(self.grade_corrections_path, delimiter=",")
#         return rows[0], np.array(rows[1:])
#
#     def process(self, path):
#         # corrections = self.read_grade_corrections()
#         corrections = [], []
#         for name in self.file_names:
#             rows = utils.read_csv_file(
#                 Path("{}/{}.tsv".format(path, name)), delimiter="\t"
#             )
#             if name == "questions_answer_key":
#                 self.parse_questions(rows)
#             else:
#                 self.parse_student_answers(rows, corrections)
#
#
# class ASAP(Dataset):
#     def __init__(self, root_path):
#         super().__init__("ASAP", ScoreEncoding.NUMERIC, extra_data=None, path=root_path)
#
#     def parse_essay_detail_description(self, question, path):
#         if question is None:
#             return
#
#         text = docx2txt.process(
#             Path(
#                 "{}/Essay_Set_Descriptions/Essay Set #{}--ReadMeFirst.docx".format(
#                     path, question.id
#                 )
#             )
#         )
#         lines = [line.strip() for line in text.split("\n") if not line.strip() == ""]
#
#         i = 0
#         start = False
#
#         while i < len(lines):
#             line = lines[i]
#             if not start and (line == "Prompt" or line == "Source Essay"):
#                 start = True
#                 question.metadata["description"] = "{}\n".format(line)
#             elif start and not line.startswith("Essay Set #") and not line.isdigit():
#                 question.metadata["description"] += "{}\n".format(line)
#             i += 1
#
#     def parse_essay_descriptions(self, path):
#         workbook = xlrd.open_workbook(
#             "{}/Essay_Set_Descriptions/essay_set_descriptions.xls".format(path)
#         )
#         sheet = workbook.sheet_by_index(0)
#         headers = []
#
#         for i in range(sheet.nrows):
#             question = None
#
#             for j in range(sheet.ncols):
#                 value = sheet.cell_value(i, j)
#                 if i == 0:
#                     headers.append(value)
#                 elif j == 0:
#                     question = Question(
#                         question_id=int(value), domain="essay", text=None
#                     )
#                 else:
#                     question.metadata[headers[j]] = value
#
#             if question is not None:
#                 self.parse_essay_detail_description(question, path)
#                 self.questions.append(question)
#
#     # ======================
#     # Note:
#     # score calculation for essay set 1, 3 - 6:
#     # domain1_score
#     #
#     # score calculation for essay set 2
#     # (domain1_score + domain2_score) / 2
#     #
#     # score calculation for essay set 7 - 8:
#     # traits are I, O, V, W, S, C
#     # formula w/o 3rd rater: (I_R1+I_R2)  +  (O_R1+O_R2)  + (S_R1+S_R2)  +  2 (C_R1+C_R2)
#     # formula w/ 3rd rater: 2 (I_R3) + 2 (O_R3) + 2 (S_R3) + 4 (C_R3)
#     # ======================
#     def parse_training_set(self, path):
#         rows = utils.read_csv_file(
#             Path("{}/training_set_rel3.tsv".format(path)), delimiter="\t"
#         )
#         headers = None
#
#         for row in rows:
#             if headers is None:
#                 headers = row
#                 continue
#
#             answer_id = int(row[0])
#             question_id = int(row[1])
#             answer_text = row[2].strip() if row[2] is not None else None
#             raw_scores = row[3:]
#             raw_scores_float = [
#                 float(score) if not score == "" else 0.0 for score in raw_scores
#             ]
#             scores = []
#
#             domain1_score = raw_scores_float[headers.index("domain1_score") - 3]
#             domain2_score = raw_scores_float[headers.index("domain2_score") - 3]
#             has_rater3 = not (row[headers.index("rater3_domain1")] == "")
#
#             # question = next((q for q in self.questions if q.id == question_id), None)
#             question = self.get_question(question_id)
#             # print('question id: {}, answer count: {}'.format(question.id, len(question.answers)))
#             if question_id == 2:
#                 scores = [(domain1_score + domain2_score) * 0.5]
#             elif question_id < 7:
#                 scores = [domain1_score]
#             elif has_rater3:
#                 rater3_traits = raw_scores_float[headers.index("rater3_trait1") - 3 :]
#                 scores = [
#                     2 * rater3_traits[0]
#                     + 2 * rater3_traits[1]
#                     + 2 * rater3_traits[4]
#                     + 4 * rater3_traits[5]
#                 ]
#             else:
#                 rater1_traits = raw_scores_float[
#                     headers.index("rater1_trait1")
#                     - 3 : headers.index("rater2_trait1")
#                     - 3
#                 ]
#                 rater2_traits = raw_scores_float[
#                     headers.index("rater2_trait1")
#                     - 3 : headers.index("rater3_trait1")
#                     - 3
#                 ]
#                 scores = [
#                     rater1_traits[0]
#                     + rater2_traits[0]
#                     + rater1_traits[1]
#                     + rater2_traits[1]
#                     + rater1_traits[4]
#                     + rater2_traits[4]
#                     + 2 * (rater1_traits[5] + rater2_traits[5])
#                 ]
#
#             question.answers.append(
#                 Answer(answer_id=answer_id, text=answer_text, scores=scores)
#             )
#             question.metadata["raw_scores"] = raw_scores
#
#     def process(self, path):
#         self.parse_essay_descriptions(path)
#         self.parse_training_set(path)
