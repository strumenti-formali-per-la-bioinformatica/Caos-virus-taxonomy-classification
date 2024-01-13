from typing import Optional
from typing import Dict
from typing import List
from typing import Any

from abc import ABCMeta
from abc import abstractmethod

import torch
import torch_geometric
import torch.nn as nn
from torch.nn import functional as F

from tabulate import tabulate
from logging import log
import numpy as np
import time
import os


class Model(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self,
                 hyperparameter: Dict[str, any],
                 weights=Optional[torch.Tensor]
                 ):
        super().__init__()
        self.__hyperparameter = hyperparameter
        self.__weights = weights

    @abstractmethod
    def step(self, batch):
        pass

    @abstractmethod
    def compute_loss(self, target: torch.Tensor, *outputs):
        pass

    @property
    def hyperparameter(self):
        return self.__hyperparameter

    def print_hyperparameter(self):
        table: List[List[str, Any]] = [[parameter, value] for parameter, value in self.__hyperparameter.items()]
        table_str: str = tabulate(
            tabular_data=table,
            headers=['hyperparameter', 'value'],
            tablefmt='psql'
        )
        return f'\n{table_str}\n'


def train(
        model: Model,
        train_loader: torch_geometric.loader.DataLoader,
        optimizer,
        model_path: str,
        device: torch.device,
        epochs: int = 10,
        model_name: str = 'model',
        evaluation: bool = False,
        val_loader: Optional[torch_geometric.loader.DataLoader] = None,
        patience: int = 10,
        logger: Optional[log] = None
) -> None:
    # print the header of the result table
    if logger is not None:
        logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | "
                    f"{'Val Loss':^10} | {'Val Acc':^9} | {'Patience':^8} | {'Elapsed':^9}")
        logger.info("-" * 80)

    # init trigger_times and last_loss for early stopping
    last_loss: float = np.inf
    trigger_times: int = 0
    best_model: Optional[Model] = None

    for epoch_i in range(epochs):
        # measure the elapsed time of each epoch
        t0_epoch: float = time.time()
        t0_batch: float = time.time()
        # reset tracking variables at the beginning of each epoch
        total_loss: float = 0
        batch_loss: float = 0
        batch_counts: int = 0

        # put the model into the training mode [IT'S JUST A FLAG]
        model.train()

        # for each batch of training data...
        for step, batch in enumerate(train_loader):
            batch_counts += 1
            # zero out any previously calculated gradients
            optimizer.zero_grad()
            # load batch to GPU
            batch = batch.to(device)

            # compute loss and accumulate the loss values
            output = model.step(batch)
            target = batch.y
            loss = model.compute_loss(target, output)
            batch_loss += loss.item()
            total_loss += loss.item()

            # perform a backward pass to calculate gradients
            loss.backward()
            # prevent the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # update parameters and the learning rate
            optimizer.step()

            # print the loss values and time elapsed for every k batches
            if (step % 50 == 0 and step != 0) or (step == len(train_loader) - 1):
                # calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # print training results
                if logger is not None:
                    logger.info(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | "
                        f"{'-':^8} | {time_elapsed:^9.2f}")

                # reset batch tracking variables
                batch_loss: float = 0
                batch_counts: int = 0
                t0_batch: float = time.time()

        # calculate the average loss over the entire training data
        avg_train_loss: float = total_loss / len(train_loader)

        if logger is not None:
            logger.info("-" * 80)

        if evaluation:
            # after the completion of each training epoch,
            # measure the model's performance on our validation set.
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            # print performance over the entire training data
            time_elapsed: float = time.time() - t0_epoch
            # early stopping
            if val_loss > last_loss:
                trigger_times += 1
            else:
                last_loss: float = val_loss
                best_model = model
                trigger_times: int = 0
            if logger is not None:
                logger.info(
                    f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | "
                    f"{val_accuracy:^9.2f} | {patience - trigger_times:^8} | {time_elapsed:^9.2f}")
                logger.info("-" * 80)
            if trigger_times >= patience:
                torch.save(best_model, os.path.join(model_path, f'{model_name}.h5'))
                break

        # save model each 5 epochs
        if epoch_i % 5 == 0 and epoch_i != 0:
            torch.save(model if best_model is None else best_model,
                       os.path.join(model_path, f'{model_name}_{epoch_i}.h5'))

    # save final model
    if best_model is not None:
        model = best_model
    torch.save(model, os.path.join(model_path, f'{model_name}.h5'))

    if logger is not None:
        logger.info("\nTraining complete!")


@torch.no_grad()
def evaluate(
        model: Model,
        val_loader: torch_geometric.loader.DataLoader,
        device: torch.device
) -> (float, float):
    # put the model into the evaluation mode.
    # the dropout layers are disabled during the test time.
    model.eval()

    # tracking variables
    val_accuracy: List[float] = []
    val_loss: List[float] = []

    # for each batch in our validation set...
    for batch in val_loader:
        # load batch to GPU
        batch = batch.to(device)
        # compute logits
        output = model.step(batch)
        target: torch.Tensor = batch.y
        # compute loss
        loss = model.compute_loss(target, output)
        val_loss.append(loss.item())
        # get the predictions
        probs: torch.Tensor = F.softmax(output[0] if isinstance(output, tuple) else output, dim=1)
        preds: torch.Tensor = torch.argmax(probs, dim=1)
        # calculate the accuracy rate
        accuracy: float = (preds == target).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # compute the average accuracy and loss over the validation set.
    val_loss: float = np.mean(val_loss)
    val_accuracy: float = np.mean(val_accuracy)

    return val_loss, val_accuracy


@torch.no_grad()
def predict(
        model: Model,
        test_loader: torch_geometric.loader.DataLoader,
        device: torch.device
) -> (np.ndarray, np.ndarray):
    # put the model into the evaluation mode. The dropout layers are disabled during the test time.
    model.eval()
    # init outputs
    outputs = []
    y_true = []
    # for each batch in our validation set...
    for batch in test_loader:
        #  load batch to GPU
        batch = batch.to(device)
        # compute logits
        output = model.step(batch)
        if isinstance(output, tuple):
            output = output[0]
        y_true.append(batch.y)
        outputs.append(output)

    # Concatenate logits from each batch
    outputs = torch.cat(outputs, dim=0)
    y_true = torch.cat(y_true, dim=0)

    # Apply softmax to calculate probabilities
    probs: np.ndarray = F.softmax(outputs, dim=1).cpu().numpy()
    y_pred: np.ndarray = np.argmax(probs, axis=1)
    y_true = y_true.cpu().numpy()

    return y_true, y_pred
