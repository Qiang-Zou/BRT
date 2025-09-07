#brt.py
from torch import nn
from torch.nn import utils as nn_utils
import torch
from . import encoders
from .encoders import PositionalEncoding
class TopoEncoder(nn.Module):
    def __init__(self,vertex_dim=3,edge_dim=128,h_dim=128,dropout=0.01):
        super(TopoEncoder, self).__init__()
        # self.edge_layer = UVNetCurveEncoder(output_dims=h_dim)
        self.ev_2_e=nn.Linear(64*3,64)
        self.wire_layer = WireNet(edge_dim)
        # self.face_layer=nn.Linear(face_dim,h_dim)
        # face_out_dim=1024
        # self.face_layer=UVNetSurfaceEncoder(output_dims=h_dim)
        # self.adj_face_layer=nn.Linear(h_dim,h_dim)
        self.adj_face_layer=nn.Sequential(
            nn.Linear(2*edge_dim,512,bias=False),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,256,bias=False),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256,h_dim),
        )

        self.last_layer=nn.Sequential(
            nn.Linear(4*edge_dim,edge_dim,bias=False),
            nn.LayerNorm(edge_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def cpu_input(self):
        return ['edge_index_length']

    def forward(self,edges,faces,edge_index,wire_index,face_index,edge_index_length,wire_index_length,adj_face_index_length):
            '''
                edges: [N_e,D_e]
                faces: [N_f,D_f]
                edge_index: [N_w,M]
                wire_index: [N_f,M]
                face_index: [N_f,M]
            '''
            N_e=edges.shape[0]
            N_w=edge_index.shape[0]
            N_f=face_index.shape[0]

            wire_mask=getMaskFromLength(wire_index_length,wire_index.shape[1])
            adj_face_mask=getMaskFromLength(adj_face_index_length,face_index.shape[1])

            ## edges=self.edge_layer(edges.permute(0,2,1))

            edges_per_wire=torch.gather(edges.unsqueeze(0).expand(N_w,-1,-1),1,
                                        edge_index.unsqueeze(-1).expand(-1,-1,edges.shape[-1])) # N_w, M, D_e

            feat_wire=self.wire_layer(edges_per_wire,edge_index_length) # N_w, D_w

            wires_per_face=torch.gather(feat_wire.unsqueeze(0).expand(N_f,-1,-1),1,
                                        wire_index.unsqueeze(-1).expand(-1,-1,feat_wire.shape[-1])) # N_f, M, D_e

            wires_per_face=torch.masked_fill(wires_per_face,~wire_mask.unsqueeze(-1),0)

            feat_face=faces
            # feat_face=torch.zeros_like(faces)

            # feat_face=self.face_layer2(feat_face)
            # torch._assert(feat_face.shape[-1]==wires_per_face.shape[-1] and feat_face.shape[0]==wires_per_face.shape[0],f'face shape error,{feat_face.shape},{wires_per_face.shape}')
            feat_face=torch.concat((feat_face,torch.sum(wires_per_face,dim=1)),dim=-1) # N_f, D_f

            adjs_faces_per_face=torch.gather(feat_face.unsqueeze(0).expand(N_f,-1,-1),1,
                                            face_index.unsqueeze(-1).expand(-1,-1,feat_face.shape[-1]))

            adjs_faces_per_face=self.adj_face_layer(adjs_faces_per_face)
            adjs_faces_per_face=torch.masked_fill(adjs_faces_per_face,~adj_face_mask.unsqueeze(-1),0)

            feat_face=torch.concat((feat_face,torch.sum(adjs_faces_per_face,dim=1)),dim=1)
            feat_face=self.last_layer(feat_face)

            return feat_face


def getMaskFromLength(length,max_length):
    mask=torch.arange(max_length).unsqueeze(0).expand(length.shape[0],-1).to(length.device) < length.unsqueeze(-1)
    return mask


class WireNet(nn.Module):
    def __init__(self,h_dim):
        super(WireNet, self).__init__()
        self.rnn=nn.RNN(h_dim, h_dim,batch_first=True)

    def forward(self,feat,feat_length):
        '''
            feat: [N_f, D_f]
            feat_length: [N_f]

            return
            last_h: [B_f]
        '''
        _,idx_sort=torch.sort(feat_length, dim=0, descending=True)
        _,idx_unsort=torch.sort(idx_sort)

        order_seq_lengths=torch.index_select(feat_length,0, idx_sort)

        idx_sort=idx_sort.to(feat.device)
        idx_unsort=idx_unsort.to(feat.device)

        order_feat=torch.index_select(feat,0, idx_sort)

        x_packed=nn_utils.rnn.pack_padded_sequence(order_feat, order_seq_lengths, batch_first=True)

        h0=torch.randn(1,feat.shape[0], feat.shape[-1]).to(feat.device)

        y_packed,h_n=self.rnn(x_packed,h0)

        last_h = torch.index_select(h_n[-1], dim=0, index=idx_unsort)

        return last_h


class BRT(nn.Module):
    def __init__(self,dmodel=256,hidden_dim=2048,n_layers=4,n_heads=16,dropout=0.01,max_face_length=90):
        super(BRT, self).__init__()
        self.edge_layer= EdgeEncoder(input_dim=4*11,srf_emb_dim=dmodel,dropout=dropout,hidden_dim=hidden_dim,n_layers=2,n_heads=4)
        self.face_layer= FaceEncoder(input_dim=28*4+1+7,srf_emb_dim=dmodel,dropout=dropout,hidden_dim=hidden_dim,n_layers=2,n_heads=4)
        # self.face_layer= FaceEncoder(input_dim=2*28*4+1,srf_emb_dim=dmodel,dropout=dropout,hidden_dim=hidden_dim,n_layers=2,n_heads=4)
        # self.face_layer= FaceEncoder_cnn(input_dim=2*4+1,srf_emb_dim=dmodel,dropout=dropout,hidden_dim=hidden_dim,n_layers=2,n_heads=4)
        self.topo_layer= TopoEncoder(vertex_dim=3,edge_dim=dmodel,h_dim=2*dmodel,dropout=dropout)
        # self.vertex_layer=VertexEncoder()
        self.transformer=encoders.TransformerEncoderBLock(
            input_dim=dmodel,c_hidden=hidden_dim,n_layers=n_layers,n_heads=n_heads,dropout=dropout)
        self.max_length=max_face_length


    def forward(self,edge,face,tri_normal,face_vis_mask,face_padding_mask,
                edge_index,wire_index,adj_face_index,edge_padding_mask,edge_index_length,wire_index_length,adj_face_index_length,
                num_faces_per_solid,**kwargs):
        
        # with torch.no_grad():
        edge_emb=self.edge_layer(edge,edge_padding_mask)
        # edge_emb=None
        # with torch.no_grad():
        face_emb=self.face_layer(face,tri_normal,face_vis_mask,face_padding_mask)
        # face_emb=self.face_layer(face,face_vis_mask,face_padding_mask)
        # face_emb=torch.zeros_like(face_emb)
        topo_emb=self.topo_layer(edge_emb,face_emb,edge_index,wire_index,adj_face_index,edge_index_length,wire_index_length,adj_face_index_length)
        topo_emb,mask=self.splitIntoBatches(topo_emb,num_faces_per_solid)
        # print(topo_emb.shape)
        perm_index=kwargs.get('perm_index',None)
        if perm_index is not None:
            perm_index=perm_index.view(-1,self.max_length)
            topo_emb=torch.gather(topo_emb,1,perm_index.unsqueeze(-1).expand(-1,-1,topo_emb.shape[-1]))[:,:kwargs['reserved_num']]
            mask=torch.gather(mask,1,perm_index)[:,:kwargs['reserved_num']]
        return self.transformer(topo_emb,torch.logical_not(mask)),mask

    def masking(self,feature,mask,max_length):
        '''
            shuffle the feature and reserve the first max_length elements
            Args:
            feature: [B, L, D]
            mask: [B, L]
            Returns:
            feature: [B, max_length, D]
            mask: [B, max_length]
        '''
        B,L,D=feature.shape
        index=torch.arange(L,dtype=mask.dtype,device=mask.device).unsqueeze(0).expand(B,-1)
        index=index.masked_fill(torch.logical_not(mask),0)
        index=index.unsqueeze(-1).expand(-1,-1,D)
        feature=torch.gather(feature,1,index)
        mask=mask[:,:max_length]
        return feature,mask

    def splitIntoBatches(self,face_features,num_faces_per_graph):
        max_width=self.max_length
        # max_width=torch.max(num_faces_per_graph)

        cell_index_per_batch=torch.cumsum(num_faces_per_graph,dim=0)
        base_index_per_batch=cell_index_per_batch-num_faces_per_graph
        index=torch.arange(max_width,dtype=face_features.dtype,device=face_features.device).unsqueeze(0).expand(len(num_faces_per_graph),-1)
        index=index+base_index_per_batch.unsqueeze(1)
        mask=index<cell_index_per_batch.unsqueeze(1).expand(-1,max_width)
        index=index.masked_fill(torch.logical_not(mask),0).long()
        batches=torch.gather(face_features.unsqueeze(0).expand(len(num_faces_per_graph),-1,-1),1,index.unsqueeze(-1).expand(-1,-1,face_features.shape[-1]))

        return batches, mask

    def BatchesIntoOneLine(self,face_features,num_faces_per_graph):
        '''
            face_features: [B, M, D]
            num_faces_per_graph: [B]
        '''
        lines=[]
        for face_feature,num_faces in zip(face_features,num_faces_per_graph):
            lines.append(face_feature[:num_faces])
        return torch.cat(lines,dim=0)

class FaceDistillationFromUV(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.face_encoder = FaceEncoder(input_dim=28*4+1,srf_emb_dim=256,dropout=0.01,hidden_dim=1024,n_layers=2,n_heads=4)
        self.uv_encoder = encoders.UVNetSurfaceEncoder(in_channels=7,output_dims=256)

    def forward(self, face, uvgrids,face_vis_mask, face_padding_mask,**kwargs):
        face_emb = self.face_encoder(face, face_vis_mask, face_padding_mask)
        with torch.no_grad():
            uv_emb = self.uv_encoder(uvgrids)
        return torch.sum((face_emb-uv_emb)**2)

class FaceEncoder(nn.Module):
    def __init__(
        self,
        input_dim=28*4,
        srf_emb_dim=64,
        dropout=0.1,
        hidden_dim=1024,
        n_layers=4,
        dislation=4,
        n_heads=16,
        num_classes=1024,
        max_face_length=600,
    ):
        """
        Initialize the UV-Net solid classification model

        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()

        # self.vis_linear=nn.Linear(input_dim+1,input_dim)
        self.surf_encoder = encoders.BezierEncoderMLP_(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=srf_emb_dim)
        # self.surf_encoder = encoders.BezierTriangleEncoder(sampled_points=64,out_dim=srf_emb_dim,vis_dim=True)

        self.transformer_encoder=encoders.TransformerEncoderBLock(
            input_dim=srf_emb_dim,c_hidden=hidden_dim,n_layers=n_layers,n_heads=n_heads,dropout=dropout)
        self.max_length=max_face_length

        # self.class_token=nn.Parameter(torch.randn(1,1,srf_emb_dim))
        self.pos=PositionalEncoding(srf_emb_dim,dropout)
        self.dislation=dislation

    def forward(self, control_pts,tri_normal, in_mask, padding_mask):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """

        mask=padding_mask

        B,L=control_pts.shape[0],control_pts.shape[1]
        x = torch.cat([torch.flatten(control_pts,start_dim=2),tri_normal,in_mask.unsqueeze(-1)],dim=-1)
        face_emb = self.surf_encoder(x.view(-1,x.shape[-1]))

        # x=control_pts.view(-1,N,D)
        # in_mask=in_mask.reshape(-1)
        # face_emb=self.surf_encoder(x,vis_mask=in_mask)

        face_emb=face_emb.view(B,L,-1)

        face_emb=self.pos(face_emb)

        # B,L,C=face_emb.shape
        # face_emb=face_emb.view(B,L//self.dislation,self.dislation,C).transpose(1,2).reshape(-1,L//self.dislation,C)
        # src_mask=mask.view(B,L//self.dislation,self.dislation).transpose(1,2).reshape(-1,L//self.dislation)

        src_mask = torch.logical_not(mask)
        face_emb=self.transformer_encoder(face_emb,src_mask)

        # face_emb=face_emb.view(B,self.dislation,L//self.dislation,-1).transpose(1,2).reshape(B,L,-1)

        face_emb=face_emb.masked_fill(torch.logical_not(mask.unsqueeze(-1)),0)
        count=mask.sum(dim=1)
        # if torch.isnan(face_emb).any():
        #     print(face_emb[0])
        #     exit(0)
        feature=face_emb.sum(dim=1)/count.unsqueeze(-1)

        return feature

class FaceEncoder_cnn(nn.Module):
    def __init__(
        self,
        input_dim=28*4,
        srf_emb_dim=64,
        dropout=0.1,
        hidden_dim=1024,
        n_layers=4,
        dislation=4,
        n_heads=16,
        num_classes=1024,
        max_face_length=600,
    ):
        """
        Initialize the UV-Net solid classification model

        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()

        # self.vis_linear=nn.Linear(input_dim+1,input_dim)
        self.surf_encoder = encoders.BezierEncoderMLP_(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=srf_emb_dim)
        # self.surf_encoder = encoders.BezierTriangleEncoder(sampled_points=64,out_dim=srf_emb_dim,vis_dim=True)
        self.cnn=encoders.UVNetSurfaceEncoder(in_channels=srf_emb_dim,output_dims=srf_emb_dim)

    def forward(self, control_pts, in_mask, padding_mask):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """

        mask=padding_mask

        B,L=control_pts.shape[0],control_pts.shape[1]
        # print(B,L)
        # test
        control_pts=control_pts[:,:,:,[0]]
        x = torch.cat([torch.flatten(control_pts,start_dim=2),in_mask.unsqueeze(-1)],dim=-1)
        face_emb = self.surf_encoder(x.view(-1,x.shape[-1]))

        # x=control_pts.view(-1,N,D)
        # in_mask=in_mask.reshape(-1)
        # face_emb=self.surf_encoder(x,vis_mask=in_mask)

        face_emb=face_emb.view(B,16,16,-1).permute(0,3,1,2)
        # print(face_emb.shape)

        face_emb=self.cnn(face_emb)

        return face_emb

class FaceEncoder_cnn2(nn.Module):
    def __init__(
        self,
        input_dim=28*4,
        srf_emb_dim=64,
        dropout=0.1,
        hidden_dim=1024,
        n_layers=4,
        dislation=4,
        n_heads=16,
        num_classes=1024,
        max_face_length=600,
    ):
        """
        Initialize the UV-Net solid classification model

        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()

        # self.vis_linear=nn.Linear(input_dim+1,input_dim)
        self.surf_encoder = encoders.BezierEncoderMLP_(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=srf_emb_dim)
        # self.surf_encoder = encoders.BezierTriangleEncoder(sampled_points=64,out_dim=srf_emb_dim,vis_dim=True)

        self.cnn=encoders.UVNetCurveEncoder(
            input_dim=srf_emb_dim,output_dims=srf_emb_dim)
        self.max_length=max_face_length

        # self.class_token=nn.Parameter(torch.randn(1,1,srf_emb_dim))
        self.pos=PositionalEncoding(srf_emb_dim,dropout)
        self.dislation=dislation

    def forward(self, control_pts,tri_normal, in_mask, padding_mask):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """

        mask=padding_mask

        B,L=control_pts.shape[0],control_pts.shape[1]
        x = torch.cat([torch.flatten(control_pts,start_dim=2),tri_normal,in_mask.unsqueeze(-1)],dim=-1)
        face_emb = self.surf_encoder(x.view(-1,x.shape[-1]))

        # x=control_pts.view(-1,N,D)
        # in_mask=in_mask.reshape(-1)
        # face_emb=self.surf_encoder(x,vis_mask=in_mask)

        face_emb=face_emb.view(B,L,-1)

        face_emb=self.pos(face_emb)

        # B,L,C=face_emb.shape
        # face_emb=face_emb.view(B,L//self.dislation,self.dislation,C).transpose(1,2).reshape(-1,L//self.dislation,C)
        # src_mask=mask.view(B,L//self.dislation,self.dislation).transpose(1,2).reshape(-1,L//self.dislation)

        src_mask = torch.logical_not(mask)
        face_emb=self.transformer_encoder(face_emb,src_mask)

        # face_emb=face_emb.view(B,self.dislation,L//self.dislation,-1).transpose(1,2).reshape(B,L,-1)

        face_emb=face_emb.masked_fill(torch.logical_not(mask.unsqueeze(-1)),0)
        count=mask.sum(dim=1)
        # if torch.isnan(face_emb).any():
        #     print(face_emb[0])
        #     exit(0)
        feature=face_emb.sum(dim=1)/count.unsqueeze(-1)

        return feature

class VertexEncoder(nn.Module):
    def __init__(
        self,
    ):
        """
        Initialize the UV-Net solid classification model

        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()

        self.mlp = encoders._MLP(num_layers=2, input_dim=3, hidden_dim=64, output_dim=64)

class EdgeEncoder(nn.Module):
    def __init__(
        self,
        input_dim=4*4,
        srf_emb_dim=64,
        dropout=0.1,
        hidden_dim=1024,
        n_layers=4,
        n_heads=16,
        num_classes=1024,
        max_face_length=600,
    ):
        """
        Initialize the UV-Net solid classification model

        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()

        self.surf_encoder = encoders.BezierEncoderMLP_(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=srf_emb_dim)

        # self.graph_encoder = encoders.UVNetGraphEncoder_No_Edge(
        #     input_dim=srf_emb_dim, output_dim=graph_emb_dim
        # )

        self.transformer_encoder=encoders.TransformerEncoderBLock(
            input_dim=srf_emb_dim,c_hidden=hidden_dim,n_layers=n_layers,n_heads=n_heads,dropout=dropout)
        # self.points_decoder = encoders.CoordinatesClassfier(
        #     feature_dim=srf_emb_dim,hidden_dim=hidden_dim,num_classes=num_classes,dropout=dropout,in_channels=1)
        # self.normal_decoder = encoders.CoordinatesClassfier(
        #     feature_dim=srf_emb_dim,num_classes=num_classes,dropout=dropout,in_channels=1)

        self.class_token=nn.Parameter(torch.randn(1,1,srf_emb_dim))
        self.pos=PositionalEncoding(srf_emb_dim,dropout)

    def forward(self,control_pts,mask):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        # print(control_pts.shape)
        # exit(0)

        x = torch.flatten(control_pts,start_dim=2)

        face_emb = self.surf_encoder(x.view(-1,x.shape[-1]))
        face_emb=face_emb.view(x.shape[0],x.shape[1],-1)

        face_emb=torch.cat([self.class_token.expand(face_emb.shape[0],-1,-1),face_emb],dim=1)
        face_emb=self.pos(face_emb)
        mask=torch.cat([torch.ones((mask.shape[0],1),dtype=mask.dtype,device=mask.device),mask],dim=1)

        src_mask = torch.logical_not(mask)
        face_emb=self.transformer_encoder(face_emb,src_mask)

        feature=face_emb[:,0]

        return feature
