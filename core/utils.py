from logging import debug, info, warning, error

import os
from pathlib import Path
import csv

import pandas as pd

from data.datasets import Answer, Dataset, Question


class InvalidNameError(Exception):
    def __init__(self, value_type, name):
        self.type = value_type
        self.name = name

    def __str__(self):
        return "INVALID {} NAME: {}".format(self.type.upper(), self.name)


def create_directories(path):
    path = os.path.normpath(path)
    parts = path.split(os.sep)

    if os.sep == "\\":
        p = Path(parts.pop(0) + os.sep)
    else:
        p = None if not path.startswith("/") else Path("/")
    for part in parts:
        p = Path(part) if p is None else p / part
        if not os.path.exists(p):
            try:
                os.makedirs(p, exist_ok=True)
            except Exception as e:
                pass
    return p


def dump_dataset(questions: list[Question], answers: list[Answer], path):
    questions_df = pd.DataFrame([vars(question) for question in questions])
    answers_df = pd.DataFrame([vars(answer) for answer in answers])

    questions_df.to_pickle(os.path.join(path, "questions.pkl"))
    answers_df.to_pickle(os.path.join(path, "answers.pkl"))


def read_csv_file(path, delimiter=","):
    rows = []
    with open(path, "r", encoding="utf8", errors="ignore") as file:
        for row in csv.reader(file, delimiter=delimiter):
            rows.append(row)
    return rows


def write_csv_file(path, file_name, rows, delimiter=","):
    path = create_directories(path)

    with open(path / file_name, "w", newline="\n") as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        for row in rows:
            writer.writerow(row)


def validate_option(options, value, name):
    if value not in options:
        raise InvalidNameError(name, value)
