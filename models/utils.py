from typing import Callable
from typing import Optional
from typing import Iterable
from typing import Union
import abc

import einops
import torch


def from_dense_batch(dense_batch: torch.Tensor, mask: torch.Tensor):
    flatten_dense_batch = einops.rearrange(dense_batch, "b s f -> (b s) f")
    flatten_mask = einops.rearrange(mask, "b s -> (b s)")
    data_x = flatten_dense_batch[flatten_mask, :]
    num_nodes = torch.sum(mask, dim=1)  # B, like 3, 4, 3
    pr_value = torch.cumsum(num_nodes, dim=0)  # B, like 3, 7, 10
    indicator_vector = torch.zeros(int(torch.sum(num_nodes, dim=0)))
    indicator_vector[pr_value[:-1]] = 1  # num_of_nodes, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1
    data_batch = torch.cumsum(indicator_vector, dim=0)  # num_of_nodes, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2
    return data_x, data_batch


class ClassificationLoss(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self._loss: Optional[Callable] = None

    def forward(self, targets: torch.Tensor, *outputs: torch.Tensor) -> torch.Tensor:
        """
        :param targets: labels
        :param outputs: predictions
        :return: loss value
        """
        outputs = outputs[0]
        loss = self._loss(outputs, targets)
        return loss

    def get_accuracy(self, targets: torch.Tensor, *outputs: torch.Tensor) -> float:
        outputs: torch.Tensor = outputs[0]
        acc = self._calculate_accuracy(outputs, targets)
        return acc

    @abc.abstractmethod
    def _get_correct(self, outputs):
        raise NotImplementedError()

    def _calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        correct = self._get_correct(outputs)
        return float(100. * (correct == targets).sum().float() / targets.size(0))


class MulticlassClassificationLoss(ClassificationLoss):
    def __init__(self, weights: Optional[Union[torch.Tensor, Iterable]] = None, reduction: Optional[str] = None,
                 label_smoothing: float = 0.0):
        super().__init__()

        if weights is None or isinstance(weights, torch.Tensor):
            self.__weights: Optional[torch.Tensor] = weights
        else:
            self.__weights: Optional[torch.Tensor] = torch.tensor(weights)

        if reduction is not None:
            self._loss: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction=reduction, weight=weights,
                                                                              label_smoothing=label_smoothing)
        else:
            self._loss: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(weight=weights,
                                                                              label_smoothing=label_smoothing)

    @property
    def weights(self) -> Optional[torch.Tensor]:
        return self.__weights

    @weights.setter
    def weights(self, weights: Optional[Union[torch.Tensor, Iterable]]):
        if weights is None or isinstance(weights, torch.Tensor):
            self.__weights: Optional[torch.Tensor] = weights
        else:
            self.__weights: Optional[torch.Tensor] = torch.tensor(weights)

    def _get_correct(self, outputs):
        return torch.argmax(outputs, dim=1)
