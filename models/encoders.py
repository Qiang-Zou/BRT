"""
    encoders whose input is group of control points of bazier pieces
"""
import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor

from typing import Callable, Union, Optional
# from models import pointnet as pn
from dgl.nn.pytorch.glob import MaxPooling
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.conv import NNConv

class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dims=64):
        """
        This is the 1D convolutional network that extracts features from the B-rep edge
        geometry described as 1D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         curve tangents. Defaults to 6.
            output_dims (int, optional): Output curve embedding dimension. Defaults to 64.
        """
        super(UVNetCurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

class UVNetSurfaceEncoder(nn.Module):
    def __init__(
        self,
        in_channels=7,
        output_dims=64,
    ):
        """
        This is the 2D convolutional network that extracts features from the B-rep face
        geometry described as 2D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         surface normals and 1 for the trimming mask. Defaults
                                         to 7.
            output_dims (int, optional): Output surface embedding dimension. Defaults to 64.
        """
        super(UVNetSurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
class UVNetSurfaceEncoder_new(nn.Module):
    def __init__(
        self,
        in_channels=5,
        hidden_dim=64,
        output_dim=64,
    ):
        """
        This is the 2D convolutional network that extracts features from the B-rep face
        geometry described as 2D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         surface normals and 1 for the trimming mask. Defaults
                                         to 7.
            output_dims (int, optional): Output surface embedding dimension. Defaults to 64.
        """
        super().__init__()
        self.dmodel = hidden_dim
        self.bezier = BezierEncoderMLP(out_dim=self.dmodel, dim=in_channels)
        # self.bezier = UVNetSurfaceEncoder(in_channels=in_channels,output_dims=self.dmodel)
        self.pos_encoder = PositionEmbeddingLearned(self.dmodel)
        # self.tranformer = nn.Transformer(
        #     d_model=output_dims, num_decoder_layers=0)
        self.transformer = MyTransformerEncoder(
            d_model=self.dmodel, dim_feedforward=128,
            num_encoder_layers=6,  # original 6
            output_dim=output_dim, dropout=0.3, nhead=4)

    def forward(self, face, pos, mask, scale):
        # face in shape [batch,len,dim,h,w]
        # pos in shape [batch,len,2]
        # mask in shape [batch,len]
        # scale in shape [batch]
        batch, len, dim, h, w = face.shape
        x = face.view(-1, dim, h, w)
        pos = pos.view(-1, 2)

        x = self.bezier(x)
        # batch*len,dmodel

        x = self.pos_encoder(x, pos)

        x = x.view(batch, len, self.dmodel)

        # x = self.transformer(src=x)
        x = self.transformer(src=x, src_key_padding_mask=mask)
        # batch,output_dims

        return x


class UVNetSurfaceEncoder2D(nn.Module):
    def __init__(
        self,
        in_channels=5,
        src_len=2*2,
        hidden_dim=64,
        output_dim=64,
    ):
        """
        Compared to UVNetSurfaceEncoder_new, use different position encoding
        """
        super().__init__()
        self.dmodel = hidden_dim
        self.bezier = BezierEncoderMLP(out_dim=self.dmodel, dim=in_channels)
        self.pos = nn.Parameter(torch.empty(
            1, src_len+1, hidden_dim).normal_(std=0.02))
        # self.tranformer = nn.Transformer(
        #     d_model=output_dims, num_decoder_layers=0)
        self.transformer = MyTransformerEncoderSimple(
            d_model=self.dmodel, dim_feedforward=128,
            num_encoder_layers=6,  # original 6
            output_dim=output_dim, dropout=0.3, nhead=4)

        self.class_token = nn.Parameter(torch.zeros(1, 1,hidden_dim))
        self.src_len=src_len

    def forward(self, face, mask, scale):
        # face in shape [batch,len,dim,h,w]
        # mask in shape [batch,len]
        # scale in shape [batch]

        if len(face.shape)==6:
            batch, len1,len2, dim, h, w = face.shape
            face=face.view(batch,-1,dim,h,w)
        if len(mask.shape)==3:
            batch,mlen1,mlen2=mask.shape
            mask=mask.view(batch,-1)
        
        batch,length, dim, h, w = face.shape
        torch._assert(self.src_len==length,'src_len not match')

        x = face.view(-1, dim, h, w)

        x = self.bezier(x)
        # batch*len,dmodel

        src = x.view(batch, -1, self.dmodel)

        src_key_padding_mask=mask

        n = batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        src = torch.cat([batch_class_token, src], dim=1)

        src = src+self.pos

        if src_key_padding_mask is not None:
            src_key_padding_mask =\
                torch.cat([torch.zeros((n, 1), dtype=torch.bool, device=src.device),
                           src_key_padding_mask], dim=1)

        x = self.transformer(src=src, src_key_padding_mask=src_key_padding_mask)

        return x


class BezierEncoderSample(nn.Module):
    def __init__(self, out_dim=64, dim=5, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.fc = _fc(dim, out_dim)

    def forward(self, x: torch.Tensor):
        # x: control pts in shape [batch,dim,h,w]
        # x: output: [batch, out_dim]
        x = x[..., 0, 0]

        x = torch.flatten(x, start_dim=1)

        return self.fc(x)


class BezierEncoderSimple(nn.Module):
    def __init__(self, out_dim=64, dim=5, patch_size=4):
        super().__init__()
        # self.dim=3
        # self.out_dim=out_dim
        # self.encoder = UVNetSurfaceEncoder(
        #     in_channels=dim, output_dims=out_dim)
        self.fc = _fc(dim*patch_size*patch_size, out_dim)

    def forward(self, x: torch.Tensor):
        # x: control pts in shape [batch,dim,h,w]
        # x: output: [batch, out_dim]

        x = torch.flatten(x, start_dim=1)

        return self.fc(x)


class BezierEncoderMLP(nn.Module):
    def __init__(self, out_dim=64, dim=5, patch_size=4, hidden_dim=256):
        super().__init__()
        # self.dim=3
        # self.out_dim=out_dim
        # self.encoder = UVNetSurfaceEncoder(
        #     in_channels=dim, output_dims=out_dim)
        # self.fc = _fc(dim*patch_size*patch_size, out_dim)
        self.mlp = _MLP(num_layers=3, input_dim=patch_size *
                        patch_size*dim, hidden_dim=hidden_dim, output_dim=out_dim)

    def forward(self, x: torch.Tensor):
        # x: control pts in shape [batch,dim,h,w]
        # x: output: [batch, out_dim]

        x = torch.flatten(x, start_dim=1)

        return self.mlp(x)

class BezierEncoderMLP2(nn.Module):
    def __init__(self, out_dim=64, dim=5, patch_size=4, hidden_dim=256):
        super().__init__()
        # self.dim=3
        # self.out_dim=out_dim
        # self.encoder = UVNetSurfaceEncoder(
        #     in_channels=dim, output_dims=out_dim)
        # self.fc = _fc(dim*patch_size*patch_size, out_dim)
        self.mlp = _MLP(num_layers=3, input_dim=patch_size *
                        patch_size*dim, hidden_dim=hidden_dim, output_dim=out_dim)
        self.mlp2 = _MLP(num_layers=3, input_dim=out_dim, hidden_dim=hidden_dim, output_dim=out_dim)

    def forward(self, x: torch.Tensor):
        # x: control pts in shape [batch,dim,h,w]
        # x: output: [batch, out_dim]

        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        x = x+self.mlp2(x)
        return x

class BezierEncoderMLP_(nn.Module):
    def __init__(self, out_dim=64, input_dim=28*4,  hidden_dim=256):
        super().__init__()
        # self.dim=3
        # self.out_dim=out_dim
        # self.encoder = UVNetSurfaceEncoder(
        #     in_channels=dim, output_dims=out_dim)
        # self.fc = _fc(dim*patch_size*patch_size, out_dim)
        self.mlp = _MLP(num_layers=3, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=out_dim)
        self.mlp2 = _MLP(num_layers=3, input_dim=out_dim, hidden_dim=hidden_dim, output_dim=out_dim)
        # self.mlp3 = _MLP(num_layers=3, input_dim=out_dim, hidden_dim=hidden_dim, output_dim=out_dim)
        # self.mlp4 = _MLP(num_layers=3, input_dim=out_dim, hidden_dim=hidden_dim, output_dim=out_dim)
        # for m in self.modules():
        #     self.weights_init(m)

    def forward(self, x: torch.Tensor):
        # x: control pts in shape [batch,dim,h,w]
        # x: output: [batch, out_dim]

        # x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        x = x+self.mlp2(x)
        # x = x+self.mlp3(x)
        # x = x+self.mlp4(x)
        return x

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)



class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding.

    Args:
        d_model: The dimension of the embedding vector.
        max_len: The maximum length of input sequences.
        dropout: The dropout probability.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model) with positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term as specified in the paper
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        # Apply sinusoidal functions to even and odd indices
        pe[:, 0::2] = torch.sin(position * div_term)    # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)    # Apply cos to odd indices

        pe = pe.unsqueeze(0)  # Shape becomes (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Register as buffer to avoid updating during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input embeddings.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model) or
               (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encodings added, same shape as input.
        """
        # Determine the shape of input x
        if x.dim() == 3:
            # Assumes shape is (batch_size, seq_len, d_model)
            x = x + self.pe[:, :x.size(1), :]
        elif x.dim() == 2:
            # Assumes shape is (seq_len, d_model)
            x = x + self.pe[:, :x.size(0), :]
        else:
            raise ValueError("Input tensor must have 2 or 3 dimensions")

        return self.dropout(x)

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    # x: batch,d_model
    # pos: batch,2

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats//2)
        self.col_embed = nn.Embedding(50, num_pos_feats//2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x, pos):
        x_emb = self.col_embed(pos[:, 0])
        y_emb = self.row_embed(pos[:, 1])
        pos = torch.cat([
            x_emb,
            y_emb
        ], dim=-1)
        return x+pos


class MyTransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None, output_dim=512):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                   activation, layer_norm_eps,
                                                   batch_first, norm_first, **factory_kwargs)
        encoder_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, encoder_norm)
        self.d_model = d_model

        self.class_token = nn.Parameter(torch.empty(1, 1, d_model))

        self.heads = _fc(d_model, output_dim)
        self.pos = nn.Parameter(torch.empty(1, 29, d_model))

        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.class_token)
        nn.init.xavier_uniform_(self.pos)

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:

        if src.size(-1) != self.d_model:
            raise RuntimeError(
                "the feature number of src must be equal to d_model")

        n = src.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        src = torch.cat([batch_class_token, src], dim=1)

        src = src+self.pos

        if src_key_padding_mask is not None:
            src_key_padding_mask =\
                torch.cat([torch.zeros((n, 1), dtype=torch.bool, device=src.device),
                           src_key_padding_mask], dim=1)

        memory = self.encoder(src, mask=mask,
                              src_key_padding_mask=src_key_padding_mask)

        x = memory[:, 0]

        x = self.heads(x)

        return x

class MyTransformerEncoderSimple(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 device=None, dtype=None, output_dim=512):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                   activation, layer_norm_eps,
                                                   batch_first, norm_first, **factory_kwargs)
        encoder_norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.d_model = d_model

        self.heads = _fc(d_model, output_dim)

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:

        if src.size(-1) != self.d_model:
            raise RuntimeError(
                "the feature number of src must be equal to d_model")

        memory = self.encoder(src, mask=mask,
                              src_key_padding_mask=src_key_padding_mask)

        x = memory[:, 0]

        x = self.heads(x)

        return x

class UVNetGraphEncoder_No_Edge(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        """
        This is the graph neural network used for message-passing features in the
        face-adjacency graph.  (see Section 3.2, Message passing in paper)

        Args:
            input_dim ([type]): [description]
            input_edge_dim ([type]): [description]
            output_dim ([type]): [description]
            hidden_dim (int, optional): [description]. Defaults to 64.
            learn_eps (bool, optional): [description]. Defaults to True.
            num_layers (int, optional): [description]. Defaults to 3.
            num_mlp_layers (int, optional): [description]. Defaults to 2.
        """
        super(UVNetGraphEncoder_No_Edge, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv_No_Edge(
                    node_feats=node_feats,
                    out_feats=hidden_dim,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                ),
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = MaxPooling()

    def forward(self, g, h):
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            # Update node features
            h = self.node_conv_layers[i](g, h)
            # Update edge features
            hidden_rep.append(h)

        out = hidden_rep[-1]
        out = self.drop1(out)
        score_over_layer = 0

        # Perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return out, score_over_layer


class _NodeConv_No_Edge(nn.Module):
    def __init__(
        self,
        node_feats,
        out_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 1 from the paper where the node features are
        updated using the neighboring node and edge features.

        Args:
            node_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(_NodeConv_No_Edge, self).__init__()

        self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)

        def apply_func(h):
            h = self.mlp(h)
            h = F.leaky_relu(self.batchnorm(h))
            return h

        self.gconv = GINConv(
            apply_func=apply_func,
            aggregator_type="sum",
            learn_eps=True
        )
        self.batchnorm = nn.BatchNorm1d(out_feats)

    def forward(self, graph, nfeat):
        return self.gconv(
            graph, nfeat)


class CoordinatesDecoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, in_channels=2, out_channels=3):
        super().__init__()
        self.uv_encoder: nn.Module = nn.Sequential(*[
            _conv1d(in_channels, hidden_dim, kernel_size=1),
            _conv1d(hidden_dim, hidden_dim, kernel_size=1),
            _conv1d(hidden_dim, feature_dim, kernel_size=1),
            ])
        self.conv1: nn.Module = _conv1d(
            feature_dim, hidden_dim, kernel_size=1)
        self.conv2: nn.Module = nn.Conv1d(
            hidden_dim, out_channels, kernel_size=1)

        self.norm=nn.BatchNorm1d(hidden_dim)
        self.relu=nn.ReLU()

    def forward(self, uv: torch.Tensor, net_feature: torch.Tensor):
        # uv (n,2,len)
        # net_feature(n,dim)

        n, _, len = uv.shape

        # n,dim,len
        x = self.uv_encoder(uv)

        # n,dim,len
        x = net_feature.unsqueeze(-1).expand(-1, -1, len) + x

        # n,hidden_dim,len
        x = self.conv1(x)

        # n,hidden_dim,len
        x = self.norm(x)

        # n,hidden_dim,len
        x = self.relu(x)

        # n,3,len
        x = self.conv2(x)

        return x.transpose(2, 1)

class CoordinatesClassfier(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, in_channels=2, out_channels=3,num_classes=1024,dropout=0.3):
        super().__init__()
        self.uv_encoder: nn.Module = _conv1d(
            in_channels, feature_dim, kernel_size=1)
        self.conv1: nn.Module = _conv1d(
            feature_dim, hidden_dim, kernel_size=1)
        self.conv2: nn.Module = nn.Conv1d(
            hidden_dim, feature_dim, kernel_size=1)

        self.classifiers=nn.ModuleList([_NonLinearClassifier(input_dim=feature_dim,num_classes=num_classes,dropout=dropout) for _ in range(out_channels)])
        # self.classifier1=_NonLinearClassifier(input_dim=feature_dim,num_classes=num_classes,dropout=dropout) 
        # self.classifier2=_NonLinearClassifier(input_dim=feature_dim,num_classes=num_classes,dropout=dropout) 
        # self.classifier3=_NonLinearClassifier(input_dim=feature_dim,num_classes=num_classes,dropout=dropout) 

        self.out_channels=out_channels

    def forward(self, uv: torch.Tensor, net_feature: torch.Tensor):
        # uv (n,2,len)
        # net_feature(n,dim)

        n, _, length = uv.shape

        # n,dim,len
        x = self.uv_encoder(uv)

        # n,dim,len
        net_feature=net_feature.unsqueeze(-1).expand(-1, -1, length)
        x = net_feature + x

        x = self.conv1(x)

        x = self.conv2(x)

        x = x+net_feature

        x = torch.transpose(x,2,1)
        x = x.reshape(n*length,-1)

        vectors = [self.classifiers[i](x) for i in range(self.out_channels)]
        # vectors = [getattr(self,f'classifier{i+1}')(x) for i in range(self.out_channels)]

        if len(vectors)==1:
            vectors= vectors[0]
        else:
            vectors = torch.stack(vectors,dim=1)
        vectors = vectors.view(n,length,self.out_channels,-1)

        return vectors
    
class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor 
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class _MLP(nn.Module):
    """"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    """
    Helper function to create a 1D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv1d, BatchNorm1d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias
        ),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    """
    Helper function to create a 2D convolutional layer with batchnorm and LeakyReLU activation

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size on each side. Defaults to 0.
        bias (bool, optional): Whether bias is used. Defaults to False.

    Returns:
        nn.Sequential: Sequential contained the Conv2d, BatchNorm2d and LeakyReLU layers
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )

def _fc_nlp(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.LayerNorm(out_features),
        nn.LeakyReLU(),
    )

class TransformerEncoderBLock(nn.Module):
    def __init__(self, input_dim, c_hidden,n_layers, n_heads, dropout=0.01,batch_first=True):
        """
        A transformer layer with a single attention layer
        """    
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(input_dim, n_heads, c_hidden, dropout,batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
    def forward(self, x,src_key_padding_mask=None,src_mask=None):
        return self.encoder(x,src_key_padding_mask=src_key_padding_mask,mask=src_mask)
