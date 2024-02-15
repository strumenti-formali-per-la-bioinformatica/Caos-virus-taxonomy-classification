from typing import Optional
from typing import Final
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from torch.nn import LayerNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.conv import GCN2Conv
from torch_geometric.nn.dense import DenseSAGEConv

NUM_CONV_LAYERS: Final = 3


class Convolutions(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 lin=True
                 ):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.drop2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)
        self.drop3 = nn.Dropout(0.5)

        if lin is True:
            self.lin = nn.Linear((NUM_CONV_LAYERS - 1) * hidden_channels + out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x1 = self.drop1(x1)
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x2 = self.drop1(x2)
        x3 = self.conv3(x2, adj, mask)
        x3 = self.drop1(x3)

        x = torch.cat([x1, x2, x3], dim=-1)

        # This is used by GNN_pool
        if self.lin is not None:
            x = self.lin(x)

        return x


class GATConvBlock(nn.Module):
    def __init__(self,
                 in_channels: Union[int, Tuple[int]],
                 out_channels: int,
                 version: str = "v2",
                 heads: int = 1,
                 concat: bool = False,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 bias: bool = True,
                 add_self_loops: bool = True,
                 edge_dim: Optional[int] = None,
                 fill_value: Union[float, Tensor, str] = 'mean',
                 project_multi_head: bool = True,
                 **kwargs):
        r"""The graph attentional operator from the `"Graph Attention Networks, paired with a normalization layer."
            <https://arxiv.org/abs/1710.10903>`_ paper
            .. math::
                \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
                \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
            where the attention coefficients :math:`\alpha_{i,j}` are computed as
            .. math::
                \alpha_{i,j} =
                \frac{
                \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
                [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
                \right)\right)}
                {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
                \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
                [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
                \right)\right)}.
            If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
            the attention coefficients :math:`\alpha_{i,j}` are computed as
            .. math::
                \alpha_{i,j} =
                \frac{
                \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
                [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
                \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
                {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
                \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
                [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
                \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.
            Args:
                in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
                    derive the size from the first input(s) to the forward method.
                    A tuple corresponds to the sizes of source and target
                    dimensionality.
                out_channels (int): Size of each output sample.
                heads (int, optional): Number of multi-head-attentions.
                    (default: :obj:`1`)
                concat (bool, optional): If set to :obj:`False`, the multi-head
                    attentions are averaged instead of concatenated.
                    (default: :obj:`True`)
                negative_slope (float, optional): LeakyReLU angle of the negative
                    slope. (default: :obj:`0.2`)
                dropout (float, optional): Dropout probability of the normalized
                    attention coefficients which exposes each node to a stochastically
                    sampled neighborhood during training. (default: :obj:`0`)
                add_self_loops (bool, optional): If set to :obj:`False`, will not add
                    self-loops to the input graph. (default: :obj:`True`)
                edge_dim (int, optional): Edge feature dimensionality (in case
                    there are any). (default: :obj:`None`)
                fill_value (float or Tensor or str, optional): The way to generate
                    edge features of self-loops (in case :obj:`edge_dim != None`).
                    If given as :obj:`float` or :class:`torch.Tensor`, edge features of
                    self-loops will be directly given by :obj:`fill_value`.
                    If given as :obj:`str`, edge features of self-loops are computed by
                    aggregating all features of edges that point to the specific node,
                    according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
                    :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
                bias (bool, optional): If set to :obj:`False`, the layer will not learn
                    an additive bias. (default: :obj:`True`)
                project_multi_head (bool, optional): If set to :obj:True`, the output of each head will be concatenated
                    and projected into an output space corresponding to out_channels.
                **kwargs (optional): Additional arguments of
                    :class:`torch_geometric.nn.conv.MessagePassing`.
            Shapes:
                - **input:**
                  node features :math:`(|\mathcal{V}|, F_{in})` or
                  :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
                  if bipartite,
                  edge indices :math:`(2, |\mathcal{E}|)`,
                  edge features :math:`(|\mathcal{E}|, D)` *(optional)*
                - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
                  :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
                  If :obj:`return_attention_weights=True`, then
                  :math:`((|\mathcal{V}|, H * F_{out}),
                  ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
                  or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
                  (|\mathcal{E}|, H)))` if bipartite
                  If :obj:`project_multi_head=True` or obj:`concat=False`, then all the previous shapes:
                  :math:`((|\mathcal{V}|, H * F_{out})`
                  will become
                  :math:`((|\mathcal{V}|, F_{out})`
            """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__version = version
        self.__heads = heads
        self.__concat = concat
        self.__dropout = dropout
        self.__bias = bias
        self.__add_self_loops = add_self_loops
        self.__edge_dim = edge_dim
        self.__fill_value = fill_value
        self.__project_multi_head = project_multi_head

        self.norm = LayerNorm(in_channels, elementwise_affine=True)

        if version == "v1":
            self.conv = GATConv(
                in_channels,
                out_channels,
                heads=heads,
                concat=concat,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=add_self_loops,
                edge_dim=edge_dim,
                fill_value=fill_value,
                bias=bias,
                **kwargs
            )
        elif version == "v2":
            self.conv = GATv2Conv(
                in_channels,
                out_channels,
                heads=heads,
                concat=concat,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=add_self_loops,
                edge_dim=edge_dim,
                fill_value=fill_value,
                bias=bias,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown version '{version}'")

        # Final projection
        self.final_linear_projection = None
        if project_multi_head and heads > 1 and concat:
            self.final_linear_projection = Linear(heads * out_channels, out_channels)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()
        if self.final_linear_projection is not None:
            self.final_linear_projection.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask

        x = self.conv(x, edge_index, edge_attr=edge_attr)

        # Apply final projection to output space if required
        if self.final_linear_projection is not None:
            x = self.final_linear_projection(x)

        return x


class GCN2ConvBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 alpha: float = 0.5,
                 theta: Optional[float] = 1.0,
                 layer: Optional[int] = 1,
                 shared_weights: bool = True,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 **kwargs):
        r"""The graph convolutional operator with initial residual connections and
            identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
            Networks" <https://arxiv.org/abs/2007.02133>`_ paper, paired with a normalization layer.
            .. math::
                \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
                \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
                \mathbf{\Theta} \right)
            with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}`, where
            :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
            matrix with inserted self-loops and
            :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
            and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
            Here, :math:`\alpha` models the strength of the initial residual
            connection, while :math:`\beta` models the strength of the identity
            mapping.
            The adjacency matrix can include other values than :obj:`1` representing
            edge weights via the optional :obj:`use_edge_weight` tensor.
            Args:
                channels (int): Size of each input and output sample.
                alpha (float): The strength of the initial residual connection
                    :math:`\alpha`.
                theta (float, optional): The hyperparameter :math:`\theta` to compute
                    the strength of the identity mapping
                    :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
                    (default: :obj:`None`)
                layer (int, optional): The layer :math:`\ell` in which this module is
                    executed. (default: :obj:`None`)
                shared_weights (bool, optional): If set to :obj:`False`, will use
                    different weight matrices for the smoothed representation and the
                    initial residual ("GCNII*"). (default: :obj:`True`)
                cached (bool, optional): If set to :obj:`True`, the layer will cache
                    the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
                    \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
                    cached version for further executions.
                    This parameter should only be set to :obj:`True` in transductive
                    learning scenarios. (default: :obj:`False`)
                normalize (bool, optional): Whether to add self-loops and apply
                    symmetric normalization. (default: :obj:`True`)
                add_self_loops (bool, optional): If set to :obj:`False`, will not add
                    self-loops to the input graph. (default: :obj:`True`)
                **kwargs (optional): Additional arguments of
                    :class:`torch_geometric.nn.conv.MessagePassing`.
            Shapes:
                - **input:**
                  node features :math:`(|\mathcal{V}|, F)`,
                  initial node features :math:`(|\mathcal{V}|, F)`,
                  edge indices :math:`(2, |\mathcal{E}|)`,
                  edge weights :math:`(|\mathcal{E}|)` *(optional)*
                - **output:** node features :math:`(|\mathcal{V}|, F)`
            """

        super().__init__()
        self.__channels = channels
        self.__alpha = alpha
        self.__theta = theta
        self.__layer = layer
        self.__shared_weights = shared_weights
        self.__cached = cached
        self.__add_self_loops = add_self_loops
        self.__normalize = normalize
        self.norm = LayerNorm(channels, elementwise_affine=True)

        self.conv = GCN2Conv(
            channels=channels,
            alpha=alpha,
            theta=theta,
            layer=layer,
            cached=cached,
            shared_weights=shared_weights,
            add_self_loops=add_self_loops,
            normalize=normalize,
            **kwargs
        )

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x: Tensor, x0: Tensor, edge_index, edge_weight, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, x0, edge_index, edge_weight=edge_weight)
