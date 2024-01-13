from typing import Optional
from typing import Dict

import torch
import einops
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from torch_geometric.utils import to_dense_batch
from models.utils import from_dense_batch

from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.pool import global_add_pool

from models import Model
from models.layers import GATConvBlock
from models.layers import GCN2ConvBlock


class UGFormerGAT(Model):
    def __init__(self,
                 hyperparameter: Dict[str, any],
                 weights=Optional[torch.Tensor]
                 ):
        super().__init__(hyperparameter, weights)

        self.__hidden_size: int = self.hyperparameter['hidden_size']
        self.__feature_dim_size: int = self.hyperparameter['dim_features']
        self.__edge_attr_dim_size: int = self.hyperparameter['dim_edge_features']
        self.__embed_dim_size = self.hyperparameter['dim_embedding']
        # each layer consists of a number of self-attention layers
        self.__n_self_att_layers: int = self.hyperparameter['att_layers']
        self.__n_layers: int = self.hyperparameter['n_layers']
        self.__n_transformer_heads = self.hyperparameter['tf_heads']
        self.__n_gat_heads = self.hyperparameter['gat_heads']
        self.__dropout: float = 0.5
        self.__n_classes: int = self.hyperparameter['n_classes']

        # project input from feature dim to embedded dim
        self.__project = Linear(
            self.__feature_dim_size,
            self.__embed_dim_size
        )

        # each layer consists of a number of self-attention layers
        # attention and convolution layers
        self.__ug_form_layers = torch.nn.ModuleList()
        self.__layers = torch.nn.ModuleList()
        self.__conv_layers = torch.nn.ModuleList()
        for _layer in range(self.__n_layers):
            encoder_layers = TransformerEncoderLayer(
                d_model=self.__embed_dim_size,
                nhead=self.__n_transformer_heads,
                dim_feedforward=self.__hidden_size,
                dropout=0.5
                # batch_first=True
            )
            # default batch_first=False (seq, batch, feature), while batch_first=True means (batch, seq, feature)
            self.__ug_form_layers.append(
                TransformerEncoder(
                    encoder_layers,
                    self.__n_self_att_layers
                )
            )
            self.__layers.append(
                GATConvBlock(
                    in_channels=self.__embed_dim_size,
                    out_channels=self.__embed_dim_size,
                    edge_dim=self.__edge_attr_dim_size,
                    dropout=self.__dropout,
                    heads=self.__n_gat_heads
                )
            )

        # linear function
        self.__predictions = torch.nn.ModuleList()
        self.__dropouts = torch.nn.ModuleList()
        for _ in range(self.__n_layers):
            self.__predictions.append(
                Linear(
                    self.__embed_dim_size,
                    self.__n_classes if self.__n_classes > 2 else 1
                )
            )
            self.__dropouts.append(
                nn.Dropout(
                    self.__dropout
                )
            )

        self.__loss = CrossEntropyLoss(weight=weights)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                *args,
                **kwargs):
        prediction_scores = 0
        input_tr = self.__project(x)
        for layer_idx in range(self.__n_layers):
            # self-Attention over all nodes
            # [batch_size, seq_length, dim] shape
            input_tr, batch_mask = to_dense_batch(x=input_tr, batch=batch, fill_value=0)
            # change tensor shape to match the transformer [seq_length, batch_size, dim]
            input_tr = einops.rearrange(input_tr, "b s f -> s b f")

            # generate attention src padding mask negating the batch_mask, because it has to be True in the padding pos
            batch_mask_transformer = batch_mask == False
            input_tr = self.__ug_form_layers[layer_idx](input_tr, src_key_padding_mask=batch_mask_transformer)

            # reshape to [batch_size, seq_length, dim] and convert to PyG batch again
            input_tr = einops.rearrange(input_tr, "s b f -> b s f")
            input_tr, _ = from_dense_batch(dense_batch=input_tr, mask=batch_mask)

            # convolution layer
            input_tr = self.__layers[layer_idx](input_tr, edge_index, *args, **kwargs)
            input_tr = F.gelu(input_tr)

            # take a sum over all node representations to get graph representations
            graph_embedding = global_add_pool(input_tr, batch=batch)
            graph_embedding = self.__dropouts[layer_idx](graph_embedding)

            # produce the final scores'
            prediction_scores += self.__predictions[layer_idx](graph_embedding)

        return prediction_scores

    def step(self, batch):
        x: torch.Tensor = batch.x.to(torch.float32)
        edge_index: torch.Tensor = batch.edge_index
        edge_attr: torch.Tensor = batch.edge_attr

        return self(x, edge_index, batch.batch, edge_attr=edge_attr)

    def compute_loss(self, target: torch.Tensor, *outputs):
        return self.__loss(outputs[0], target)


class UGFormerGCN(Model):
    def __init__(self,
                 hyperparameter: Dict[str, any],
                 weights=Optional[torch.Tensor]
                 ):
        super().__init__(hyperparameter, weights)

        self.__hidden_size: int = self.hyperparameter['hidden_size']
        self.__feature_dim_size: int = self.hyperparameter['dim_features']
        self.__edge_attr_dim_size: int = self.hyperparameter['dim_edge_features']
        self.__embed_dim_size: int = self.hyperparameter['dim_embedding']
        # each layer consists of a number of self-attention layers
        self.__n_self_att_layers: int = self.hyperparameter['att_layers']
        self.__n_layers: int = self.hyperparameter['n_layers']
        self.__n_transformer_heads: int = self.hyperparameter['tf_heads']
        self.__dropout: float = 0.5
        self.__n_classes: int = self.hyperparameter['n_classes']

        # project input from feature dim to embedded dim
        self.__project = Linear(
            self.__feature_dim_size,
            self.__embed_dim_size
        )

        # each layer consists of a number of self-attention layers
        # attention and convolution layers
        self.__ug_form_layers = torch.nn.ModuleList()
        self.__layers = torch.nn.ModuleList()
        for _layer in range(self.__n_layers):
            encoder_layers = TransformerEncoderLayer(
                d_model=self.__embed_dim_size,
                nhead=self.__n_transformer_heads,
                dim_feedforward=self.__hidden_size,
                dropout=0.5
                # batch_first=True
            )
            # default batch_first=False (seq, batch, feature), while batch_first=True means (batch, seq, feature)
            self.__ug_form_layers.append(
                TransformerEncoder(
                    encoder_layers,
                    self.__n_self_att_layers
                )
            )
            self.__layers.append(
                GCN2ConvBlock(
                    channels=self.__embed_dim_size,
                    layer=_layer + 1
                )
            )

        # linear function
        self.__predictions = torch.nn.ModuleList()
        self.__dropouts = torch.nn.ModuleList()
        for _ in range(self.__n_layers):
            self.__predictions.append(
                Linear(
                    self.__embed_dim_size,
                    self.__n_classes if self.__n_classes > 2 else 1
                )
            )
            self.__dropouts.append(
                nn.Dropout(
                    self.__dropout
                )
            )

        self.__loss = CrossEntropyLoss(weight=weights)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                *args,
                **kwargs):
        prediction_scores = 0
        input_tr = self.__project(x)
        x_0 = input_tr
        for layer_idx in range(self.__n_layers):
            # self-Attention over all nodes
            # [batch_size, seq_length, dim] shape
            input_tr, batch_mask = to_dense_batch(x=input_tr, batch=batch, fill_value=0)
            # change tensor shape to match the transformer [seq_length, batch_size, dim]
            input_tr = einops.rearrange(input_tr, "b s f -> s b f")

            # generate attention src padding mask negating the batch_mask, because it has to be True in the padding pos
            batch_mask_transformer = batch_mask == False
            input_tr = self.__ug_form_layers[layer_idx](input_tr, src_key_padding_mask=batch_mask_transformer)

            # reshape to [batch_size, seq_length, dim] and convert to PyG batch again
            input_tr = einops.rearrange(input_tr, "s b f -> b s f")
            input_tr, _ = from_dense_batch(dense_batch=input_tr, mask=batch_mask)

            # convolution layer
            input_tr = self.__layers[layer_idx](input_tr, x_0, edge_index, kwargs['edge_attr'].T[-1].to(torch.float32))
            input_tr = F.gelu(input_tr)

            # take a sum over all node representations to get graph representations
            graph_embedding = global_add_pool(input_tr, batch=batch)
            graph_embedding = self.__dropouts[layer_idx](graph_embedding)

            # produce the final scores'
            prediction_scores += self.__predictions[layer_idx](graph_embedding)

        return prediction_scores

    def step(self, batch):
        x: torch.Tensor = batch.x.to(torch.float32)
        edge_index: torch.Tensor = batch.edge_index
        edge_attr: torch.Tensor = batch.edge_attr

        return self(x, edge_index, batch.batch, edge_attr=edge_attr)

    def compute_loss(self, target: torch.Tensor, *outputs):
        return self.__loss(outputs[0], target)