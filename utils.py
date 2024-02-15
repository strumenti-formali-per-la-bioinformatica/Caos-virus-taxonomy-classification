from typing import Final
from typing import List
from typing import Dict
from typing import Any

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd
import logging
import shutil
import os

SEPARATOR: Final = '\n------------------------------------\n'


def create_test_name(len_read: int, len_overlap: int, k_size: int, hyperparameter: Dict[str, Any]) -> str:
    test_name: str = f'{len_read}_{len_overlap}_{k_size}'
    for parameter in hyperparameter.keys():
        test_name += f'_{parameter}_{hyperparameter[parameter]}'

    return test_name


def test_check(model_name: str, parent_name: str) -> bool:
    log_path = os.path.join(os.getcwd(), 'log', model_name, parent_name)
    if os.path.exists(log_path):
        model_path = os.path.join(log_path, 'model', 'model.h5')
        if os.path.exists(model_path):
            return True
        else:
            shutil.rmtree(log_path)
            return False
    else:
        return False


def create_folders(model_name: str, parent_name: str):
    # create log folder
    log_path = os.path.join(os.getcwd(), 'log', model_name, parent_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # create model folder
    model_path = os.path.join(log_path, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return log_path, model_path


def setup_logger(name, file_path, level=logging.INFO):
    handler = logging.FileHandler(file_path)
    handler.setFormatter(logging.Formatter('%(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    return logger


def close_loggers(loggers: List[logging.Logger]):
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def save_result(
        result_csv_path: str,
        len_read: int,
        len_overlap: int,
        hyperparameter: Dict[str, Any],
        y_true: List[int],
        y_pred: List[int]
):
    # init columns of result df
    columns = ['len_read', 'len_overlap']
    columns += list(hyperparameter.keys())
    columns += ['accuracy', 'precision', 'recall', 'f1-score']

    # create row of df
    values = [len_read, len_overlap]
    values += [hyperparameter[p] for p in hyperparameter.keys()]
    accuracy = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='weighted',
        zero_division=1
    )
    values += [accuracy, precision, recall, f_score]
    result_csv: pd.DataFrame = pd.DataFrame(
        [
            values
        ],
        columns=columns
    )

    # check if result dataset exists
    if os.path.exists(result_csv_path):
        global_results_csv: pd.DataFrame = pd.read_csv(result_csv_path)
        global_results_csv = pd.concat([global_results_csv, result_csv])
        global_results_csv = global_results_csv.sort_values(by=['accuracy'], ascending=False)
        global_results_csv.to_csv(result_csv_path, index=False)
    else:
        result_csv.to_csv(result_csv_path, index=False)
