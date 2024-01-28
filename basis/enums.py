from enum import Enum


class ScoreEncoding(Enum):
    TEXT = 0
    NUMERIC = 1
    UNKNOWN = 2


class TrainingType(Enum):
    UNSPECIFIED = 0
    TRAINING = 1
    TESTING_UNSEEN_ANSWERS = 2
    TESTING_UNSEEN_DOMAINS = 3
    TESTING_UNSEEN_QUESTIONS = 4


class ReferenceAnswerCategory(Enum):
    UNSPECIFIED = 0
    MINIMAL = 1
    GOOD = 2
    BEST = 3
