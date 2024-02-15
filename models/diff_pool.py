# This implementation is based on the one from the repository:
# https://github.com/diningphil/gnn-comparison,
# all rights reserved to authors and contributors.
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from typing import Optional
from typing import Dict
from typing import Any

from math import ceil

import torch
import torch.nn as nn
from torch.nn import functional as F

from models import Model
from models.layers import Convolutions
from models.layers import NUM_CONV_LAYERS

from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.dense import dense_diff_pool

from models.utils import MulticlassClassificationLoss


class DiffPoolMulticlassClassificationLoss(MulticlassClassificationLoss):
    """
    DiffPool - No Link Prediction Loss, that one is outputed by the DiffPool layer
    """

    def forward(self, targets: torch.Tensor, *outputs: torch.Tensor) -> torch.Tensor:
        if len(outputs) == 1:
            outputs = outputs[0]
        preds, lp_loss, ent_loss = outputs

        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        loss = self._loss(preds, targets)
        return loss + lp_loss + ent_loss


class DiffPoolLayer(nn.Module):
    """
    Applies GraphSAGE convolutions and then performs pooling
    """

    def __init__(self, dim_input, dim_hidden, dim_embedding, no_new_clusters):
        """
        :param dim_input:
        :param dim_hidden: embedding size of first 2 SAGE convolutions
        :param dim_embedding: embedding size of 3rd SAGE convolutions (eq. 5, dim of Z)
        :param no_new_clusters: number of clusters after pooling (eq. 6, dim of S)
        """
        super().__init__()
        self.gnn_pool = Convolutions(dim_input, dim_hidden, no_new_clusters)
        self.gnn_embed = Convolutions(dim_input, dim_hidden, dim_embedding, lin=False)

    def forward(self, x, adj, mask=None):
        s = self.gnn_pool(x, adj, mask)
        x = self.gnn_embed(x, adj, mask)

        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e


class DiffPool(Model):
    def __init__(self,
                 hyperparameter: Dict[str, Any],
                 weights: Optional[torch.Tensor] = None
                 ):
        super().__init__(hyperparameter, weights)

        self.__dim_features: int = self.hyperparameter['dim_features']
        self.__max_num_nodes: int = self.hyperparameter['max_num_nodes']
        self.__num_layers: int = self.hyperparameter['n_layers']
        self.__gnn_dim_hidden: int = self.hyperparameter['hidden_size']
        self.__dim_embedding: int = self.hyperparameter['dim_embedding']
        self.__dim_embedding_mlp: int = self.hyperparameter['dim_embedding_mlp']
        self.__n_classes: int = self.hyperparameter['n_classes']

        # reproduce paper choice about coarse factor
        coarse_factor = 0.1 if self.__num_layers == 1 else 0.25

        gnn_dim_input = self.__dim_features
        no_new_clusters = ceil(coarse_factor * self.__max_num_nodes)
        gnn_embed_dim_output = (NUM_CONV_LAYERS - 1) * self.__gnn_dim_hidden + self.__dim_embedding

        layers = []
        for i in range(self.__num_layers):
            diff_pool_layer = DiffPoolLayer(gnn_dim_input, self.__gnn_dim_hidden, self.__dim_embedding, no_new_clusters)
            layers.append(diff_pool_layer)
            # update embedding sizes
            gnn_dim_input = gnn_embed_dim_output
            no_new_clusters = ceil(no_new_clusters * coarse_factor)
        self.__diff_pool_layers = nn.ModuleList(layers)

        # after DiffPool layers, apply again layers of GraphSAGE convolutions
        self.__final_embed = Convolutions(gnn_embed_dim_output, self.__gnn_dim_hidden, self.__dim_embedding, lin=False)
        final_embed_dim_output = gnn_embed_dim_output * (self.__num_layers + 1)

        self.__lin1 = nn.Linear(final_embed_dim_output, self.__dim_embedding_mlp)
        self.__drop = nn.Dropout(0.5)
        self.__lin2 = nn.Linear(self.__dim_embedding_mlp, self.__n_classes)

        self.__loss = DiffPoolMulticlassClassificationLoss(weights=weights)

    def forward(self, x, edge_index, batch):
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)

        # adj, mask, x = data.adj, data.mask, data.x
        x_all, l_total, e_total = [], 0, 0

        for i in range(self.__num_layers):
            if i != 0:
                mask = None
            x, adj, l, e = self.__diff_pool_layers[i](x, adj, mask)  # x has shape (batch, MAX_no_nodes, feature_size)
            x_all.append(torch.max(x, dim=1)[0])
            l_total += l
            e_total += e

        x = self.__final_embed(x, adj)
        x_all.append(torch.max(x, dim=1)[0])

        x = torch.cat(x_all, dim=1)  # shape (batch, feature_size x diff_pool layers)
        x = F.relu(self.__lin1(x))
        x = self.__drop(x)
        x = self.__lin2(x)

        return x, l_total, e_total

    def step(self, batch):
        x: torch.Tensor = batch.x.to(torch.float32)
        edge_index: torch.Tensor = batch.edge_index

        return self(x, edge_index, batch.batch)

    def compute_loss(self, target: torch.Tensor, *outputs):
        return self.__loss(target, *outputs)
