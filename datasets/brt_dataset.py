#brt dataset
import random
import numpy as np
import dgl
from dgl.data.utils import load_graphs
import json
from torch.utils.data import Dataset, DataLoader
import pathlib
from tqdm import tqdm
import torch
import logging
logger=logging.getLogger('dataset loading')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/dataset.log', mode='w')
logger.addHandler(file_handler)
# from dataset.txtEmbedding import getTextEmbeddings,MFCADPP_SEG_LABELS

class BRTDataset(Dataset):
    def load_samples(self, items, center_and_scale=False):
        '''
            item: {'topo_path':<file_path>,'label':<label>,'face_path':<file_path>,'edge_path':<file_path>}
        '''
        self.data = []
        self.labels = []
        for item in tqdm(items):
            sample = self.load_one_sample(item)
            if sample is not None:
                sample = self.normalize(sample)
                sample = self.padding(sample)
                sample = self.convert_to_float32(sample)
                self.data.append(sample)
        logger.debug(self.data[0].keys())

    def convert_to_float32(self, data):
        '''
            data: {'face':[],'face_vis_mask':[],'face_points':[],'edge':[],'topo':[],'label':[]}
        '''
        new_data={}
        for k,v in data.items():
            # print(k,v.type())
            if k in ['face','edge','tri_normal']:
                new_data[k]=v.type(torch.FloatTensor)
            else:
                new_data[k]=v

        return new_data

    def padding(self,data,max_facet_len=100,max_arc_len=50,padding_mode='zero'):
        '''
            data: {'face':[],'face_vis_mask':[],'face_points':[],'edge':[],'topo':[],'label':[]}
        '''
        faces_padded=[]
        tri_normals_padded=[]
        in_masks_padded=[]
        padding_masks=[]
        faces=data['face']
        in_masks=data['face_vis_mask']
        tri_normals=data['tri_normal']

        for nodes,tri_normal,in_mask in zip(faces,tri_normals,in_masks):
            # assert len(nodes)==len(in_mask)
            # nodes=nodes[in_mask]
            # in_mask=torch.ones_like(in_mask)[:nodes.shape[0]]
            if nodes.shape[0]>max_facet_len:
                # randomly select max_facet_len faces
                idxs=torch.randperm(nodes.size(0))[:max_facet_len]
                nodes=nodes[idxs]
                in_mask=in_mask[idxs]
                # print(idxs.shape)
                # print(tri_normal.shape)
                tri_normal=tri_normal[idxs]

            # nodes=nodes[:max_facet_len]
            # in_mask=in_mask[:max_facet_len]

            len_nodes=nodes.shape[0]
            if len_nodes==max_facet_len:
                nodes_padded=nodes
                in_mask_padded=in_mask
                tri_normal_padded=tri_normal
                padding_mask=torch.ones(max_facet_len,dtype=torch.bool)
            else:
                if padding_mode=='zero':
                    nodes_padded=torch.zeros(max_facet_len,*nodes.shape[1:],dtype=nodes.dtype)
                    nodes_padded[:len_nodes]=nodes

                    tri_normal_padded=torch.zeros(max_facet_len,*tri_normal.shape[1:],dtype=nodes.dtype)
                    tri_normal_padded[:len_nodes]=tri_normal

                    in_mask_padded=torch.zeros(max_facet_len,dtype=in_mask.dtype)
                    in_mask_padded[:len_nodes]=in_mask
                elif padding_mode=='circular':
                    nodes_padded=nodes
                    in_mask_padded=in_mask
                    tri_normal_padded=tri_normal
                    while nodes_padded.shape[0]<max_facet_len:
                        len_nodes_padded=nodes_padded.shape[0]
                        nodes_padded=torch.cat([nodes_padded,nodes[:max_facet_len-len_nodes_padded]],dim=0)
                        tri_normal_padded=torch.cat([tri_normal_padded,tri_normal[:max_facet_len-len_nodes_padded]],dim=0)
                        in_mask_padded=torch.cat([in_mask_padded,in_mask[:max_facet_len-len_nodes_padded]],dim=0)
                        # print(nodes_padded.shape,in_mask_padded.shape)
                    # print('circular padding')
                    
                padding_mask=torch.zeros(max_facet_len,dtype=torch.bool)
                padding_mask[:len_nodes]=1

            faces_padded.append(nodes_padded)
            in_masks_padded.append(in_mask_padded)
            padding_masks.append(padding_mask)
            tri_normals_padded.append(tri_normal_padded)

        data['face']=torch.stack(faces_padded)
        data['face_vis_mask']=torch.stack(in_masks_padded)
        data['face_padding_mask']=torch.stack(padding_masks)
        data['tri_normal']=torch.stack(tri_normals_padded)
        # padding edges
        edges_padded=[]
        edges=data['edge']
        padding_masks=[]
        for edge in edges:
            if edge.shape[0]>max_arc_len:
                # randomly select max_facet_len faces
                idxs=torch.randperm(edge.size(0))[:max_arc_len]
                edge=edge[idxs]

            len_edge=edge.shape[0]
            edge_padded=torch.zeros(max_arc_len,edge.shape[-2],edge.shape[-1],dtype=edge.dtype)
            padding_mask=torch.zeros(max_arc_len,dtype=torch.bool)
            edge_padded[:len_edge]=edge
            padding_mask[:len_edge]=1
            edges_padded.append(edge_padded)
            padding_masks.append(padding_mask)
        
        data['edge']=torch.stack(edges_padded)
        data['edge_padding_mask']=torch.stack(padding_masks)

        return data
    def checkFacetLength(self, data):
        '''
            data: {'face':[],'face_vis_mask':[],'face_points':[],'edge':[],'topo':[],'label':[]}
        '''
        return True
        faces=data['face']
        for face in faces:
            if face.shape[0]>800 or face.shape[0]<=0:
                # logger.debug('face length error'+','+str(face.shape[0]))
                return False
        
        # if len(edges)==0:
        #     return False
        return True
    def normalize(self, data):
        '''
            data: {'face':[],'face_vis_mask':[],'face_points':[],'edge':[],'topo':[],'label':[]}
        '''
        points=data['points'] # (N_f,256,3)
        mean = torch.mean(points, dim=(0,1))
        points -= mean.unsqueeze(0).unsqueeze(0)
        scale=torch.max(torch.abs(points))
        if scale<1e-7:
            scale=1
        faces=data['face']

        # for face in faces:
            # face[...,:3]-=mean
            # face[...,:3]/=scale

            # for f in face:
            #     if torch.sum(f[:,-1])<1e-6:
            #         f[:,-1]=1
            #     else:
            #         f[:,-1]/=torch.sum(f[:,-1])

        edges=[torch.from_numpy(edge) for edge in data['edge']]
        for edge in edges:
            edge[...,:3]-=mean
            edge[...,:3]/=scale

        # if torch.isnan(nodes).any() or torch.isnan(sampled_points).any():
        #     raise ValueError('nan in nodes')

        data['edge']=edges
        data['face']=faces

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _collate(self, batch):
        new_data={}
        for data in batch:
            for k,v in data.items():
                if k not in new_data:
                    new_data[k]=[]
                new_data[k].append(v)

        face_num=[face.shape[0] for face in new_data['face']]
        face_num=torch.tensor(face_num)
        face_num_csum=torch.cumsum(face_num,dim=0)-face_num

        adj_index_index=new_data['adj_face_index']
        for i in range(len(adj_index_index)):
            adj_index_index[i]=adj_index_index[i]+face_num_csum[i]

        edge_num=[edge.shape[0] for edge in new_data['edge']]
        edge_num=torch.tensor(edge_num)
        edge_num_csum=torch.cumsum(edge_num,dim=0)-edge_num

        edge_index=new_data['edge_index']
        for i in range(len(edge_index)):
            edge_index[i]=edge_index[i]+edge_num_csum[i]

        wire_num=[wire.shape[0] for wire in new_data['edge_index']]
        wire_num=torch.tensor(wire_num)
        wire_num_csum=torch.cumsum(wire_num,dim=0)-wire_num

        wire_index=new_data['wire_index']
        for i in range(len(wire_index)):
            wire_index[i]=wire_index[i]+wire_num_csum[i]

        # for k,v in new_data.items():
        #     print(k)
        #     print(type(v[0]))
        new_data={k:(torch.cat(v) if k not in ['label','filename'] else (v if k=='filename' else torch.tensor(v)) ) for k,v  in new_data.items()}

        # new_data={k:torch.cat(v) if k not in ['filename'] else v for k,v in new_data.items() }
        # new_data={k:torch.cat(v) for k,v  in new_data.items()}
        new_data['num_faces_per_solid']=face_num.long()
        # new_data['label']=torch.flatten(new_data['label'])

        return new_data

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,  # Can be set to non-zero on Linux
            drop_last=True,
        )

    def load_one_sample(self, item):
        # Load the graph using base class method
        """
        Returns:
            sample: token_update arrays
            label: [{'instantces':[],'embedding':<txt_embedding>} for each feature]
        """
        face = self.load_face(item['face'])
        # edge = self.load_edge(item['edge_path'])
        topo = self.load_topo(item['topo'])

        if face is None or topo is None:
            # logger.debug('None face or topo'+','+str(face is None)+','+str(topo is None))
            return None

        label = item['label']

        data={}

        # data['face']=face
        # data['topo']=topo
        data.update(face)
        data.update(topo)
        data['label']=label

        return data

    def __init__(
        self, root_dir, split="train", center_and_scale=True, random_rotate=False,
        masking_rate=None,masking_rate_v2=None,load_label_from_file=None
    ):
        """
        Args:
            root_dir (str): Root path of dataset
            split (str, optional): Data split to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        path = pathlib.Path(root_dir)
        self.path = path
        assert split in ("train", "val", "test")
        # if split=='train':
        #     split='test'

        data_dir = path.joinpath('datasplit.json')
        filelist=[]

        with open(data_dir, "r") as f:
            # if split=='test':
            #     split='val'
            data = json.load(f)[split]
            for item in data:
                filelist.append(item)

        self.random_rotate = random_rotate
        self.masking_rate=masking_rate
        self.masking_rate_v2=masking_rate_v2
        self.load_label_from_file=load_label_from_file

        all_files = filelist

        # Load graphs
        print(f"Loading {split} data...")
        # self.load_samples(all_files, center_and_scale)
        self.load_samples(all_files, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_face(self, file_path):
        '''
            return list of faces, each element is {'nodes','in_mask'}
        '''
        labels = torch.load(file_path)
        data= {'face':labels['nodes'],'face_vis_mask':labels['in_mask'],'points':labels['points'],'tri_normal':labels['tri_normals']} 
        if not self.checkFacetLength(data):
            return None
        return data
        # face_num=len(labels['nodes'])
        # return [{'nodes': labels['nodes'][i],'in_mask': labels['in_mask'][i], 'points': labels['points'][i]} for i in range(face_num)]

    def load_edge(self, file_path):
        '''
            return list of edges, each element is ctrl_points
        '''
        labels = torch.load(file_path)
        return labels
        # face_num=len(labels['edge'])
        # return [labels['edge'][i] for i in range(face_num)]

    def load_topo(self, file_path):
        solid = torch.load(file_path)

        edge_index=solid['edge_index']
        wire_index=solid['wire_index']
        adj_face_index=solid['adj_face_index']

        max_face_index_length=30
        max_wire_index_length=10
        max_edge_index_length=30
        max_total_face_length=90

        edge_index_length=[]
        if len(wire_index)>max_total_face_length:
            # raise RuntimeError('len(edge_index)>max_total_face_length')
            logger.warning('len(edge_index)>max_total_face_length')
            return None
        wire_index_length=torch.zeros(len(wire_index),dtype=torch.long)
        adj_face_index_length=torch.zeros(len(adj_face_index),dtype=torch.long)


        # adj_face_index is a list, each item in the list is the ajdacent face indices of one face
        for fid,index_lst in enumerate(adj_face_index):
            adj_face_index_length[fid]=len(index_lst)
            if len(index_lst)>max_face_index_length:
                # raise RuntimeError('adj_face_index_length[fid]>max_face_index_length')
                logger.warning('adj_face_index_length[fid]>max_face_index_length')
                return None
            while len(index_lst)<max_face_index_length:
                index_lst.append(0)

        # wire_index is a list, each item in the list is the wire indices of one face
        for fid,index_lst in enumerate(wire_index):
            wire_index_length[fid]=len(index_lst)
            if len(index_lst)>max_wire_index_length:
                # raise RuntimeError('wire_index_length[fid]>max_wire_index_length')
                logger.warning('wire_index_length[fid]>max_wire_index_length')
                return None
            while len(index_lst)<max_wire_index_length:
                index_lst.append(0)

        # edge_index is a list, each item in the list is the edge indices of one wire
        for wid,index_lst in enumerate(edge_index):
            edge_index_length.append(len(index_lst))
            if len(index_lst)>max_edge_index_length:
                # raise RuntimeError('edge_index_length[fid]>max_edge_index_length')
                logger.warning('edge_index_length[fid]>max_edge_index_length')
                return None
            while len(index_lst)<max_edge_index_length:
                index_lst.append(0)

        edge_index_length=torch.tensor(edge_index_length,dtype=torch.long)
        edge_index=torch.tensor(edge_index,dtype=torch.long)
        adj_face_index=torch.tensor(adj_face_index,dtype=torch.long)
        wire_index=torch.tensor(wire_index,dtype=torch.long)

        return {
            'edge':solid['edge'],
            'edge_index_length':edge_index_length,'wire_index_length':wire_index_length,
                'adj_face_index_length':adj_face_index_length,
                'edge_index':edge_index,'wire_index':wire_index,'adj_face_index':adj_face_index}

class BRTDataset_seg_online(BRTDataset):
    def load_topo(self, file_path,label_path=None,masking_rate=None):
        solid = torch.load(file_path)

        if label_path is not None and type(label_path)==str:
            label=np.loadtxt(label_path)
            if len(label.shape)==0:
                label=np.array([label])
            label=torch.from_numpy(label).long()
        else:
            label=solid['label']

        edge_index=solid['edge_index']
        wire_index=solid['wire_index']
        adj_face_index=solid['adj_face_index']

        max_face_index_length=10
        max_wire_index_length=30
        max_edge_index_length=10
        max_total_face_length=100

        edge_index_length=[]

        # print(label_path)
        # print(label.shape)
        # torch._assert(len(wire_index)==len(label),f'{len(wire_index)}!={len(label),{label_path}}')

        if len(wire_index)>max_total_face_length:
            # raise RuntimeError('len(edge_index)>max_total_face_length')
            logger.warning('len(edge_index)>max_total_face_length')
            return None

        if masking_rate is not None:
            reserved_num=int(max_total_face_length*masking_rate)
            perm_index=torch.randperm(len(wire_index),dtype=torch.long)
            perm_index_padding=torch.arange(max_total_face_length,dtype=torch.long)
            perm_index_padding[:len(wire_index)]=perm_index
            label=label[perm_index][:reserved_num]
            perm_index=perm_index_padding
        else:
            perm_index=None
            reserved_num=0

        wire_index_length=torch.zeros(len(wire_index),dtype=torch.long)
        adj_face_index_length=torch.zeros(len(adj_face_index),dtype=torch.long)

        for fid,index_lst in enumerate(adj_face_index):
            if len(index_lst)>max_face_index_length:
                # logger.warning('adj_face_index_length[fid]>max_face_index_length')
                # return None
                index_lst=random.sample(index_lst,max_face_index_length)
                adj_face_index[fid]=index_lst
            adj_face_index_length[fid]=len(index_lst)
            while len(index_lst)<max_face_index_length:
                index_lst.append(0)

        # wire_index is a list, each item in the list is the wire indices of one face
        for fid,index_lst in enumerate(wire_index):
            if len(index_lst)>max_wire_index_length:
                # logger.warning('wire_index_length[fid]>max_wire_index_length')
                # return None
                index_lst=random.sample(index_lst,max_wire_index_length)
                wire_index[fid]=index_lst
            wire_index_length[fid]=len(index_lst)
            while len(index_lst)<max_wire_index_length:
                index_lst.append(0)

        # edge_index is a list, each item in the list is the edge indices of one wire
        for wid,index_lst in enumerate(edge_index):
            if len(index_lst)>max_edge_index_length:
                # logger.warning('edge_index_length[fid]>max_edge_index_length')
                # return None
                index_lst=random.sample(index_lst,max_edge_index_length)
                edge_index[wid]=index_lst # is this In-Place Modification safe? Answer: Yes
            edge_index_length.append(len(index_lst))
            while len(index_lst)<max_edge_index_length:
                index_lst.append(0)

        edge_index_length=torch.tensor(edge_index_length,dtype=torch.long)
        edge_index=torch.tensor(edge_index,dtype=torch.long)
        if type(adj_face_index)==list:
            adj_face_index=torch.tensor(adj_face_index,dtype=torch.long)
        if type(wire_index)==list:
            wire_index=torch.tensor(wire_index,dtype=torch.long)

        return {
            'label':label,
            'edge':solid['edge'],
            'edge_index_length':edge_index_length,'wire_index_length':wire_index_length,
                'adj_face_index_length':adj_face_index_length,
                'edge_index':edge_index,'wire_index':wire_index,'adj_face_index':adj_face_index},perm_index
    def load_one_sample(self, item):
        # Load the graph using base class method
        """
        Returns:
            sample: token_update arrays
            label: [{'instantces':[],'embedding':<txt_embedding>} for each feature]
        """
        face = self.load_face(item['face'])
        # edge = self.load_edge(item['edge_path'])
        if self.load_label_from_file:
            topo= self.load_topo(item['topo'],item['label'],self.masking_rate)
        else:
            topo = self.load_topo(item['topo'],masking_rate=self.masking_rate)

        if face is None or topo is None:
            # logger.debug('None face or topo'+','+str(face is None)+','+str(topo is None))
            return None
        data={}
        topo,perm_index=topo

        if self.masking_rate is not None:
            data['perm_index']=perm_index
        # data['reserved_num']=reserved_num

        data.update(face)
        data.update(topo)

        data['filename']=item['face']

        return data

    def _collate(self, batch):
        new_data={}
        for data in batch:
            for k,v in data.items():
                if k not in new_data:
                    new_data[k]=[]
                new_data[k].append(v)

        face_num=[face.shape[0] for face in new_data['face']]
        face_num=torch.tensor(face_num)
        face_num_csum=torch.cumsum(face_num,dim=0)-face_num

        adj_index_index=new_data['adj_face_index']
        for i in range(len(adj_index_index)):
            adj_index_index[i]=adj_index_index[i]+face_num_csum[i]

        edge_num=[edge.shape[0] for edge in new_data['edge']]
        edge_num=torch.tensor(edge_num)
        edge_num_csum=torch.cumsum(edge_num,dim=0)-edge_num

        edge_index=new_data['edge_index']
        for i in range(len(edge_index)):
            edge_index[i]=edge_index[i]+edge_num_csum[i]

        wire_num=[wire.shape[0] for wire in new_data['edge_index']]
        wire_num=torch.tensor(wire_num)
        wire_num_csum=torch.cumsum(wire_num,dim=0)-wire_num

        wire_index=new_data['wire_index']
        for i in range(len(wire_index)):
            wire_index[i]=wire_index[i]+wire_num_csum[i]

        # for k,v in new_data.items():
        #     print(k)
        #     print(type(v[0]))
        new_data={k:torch.cat(v) if k not in ['filename'] else v for k,v in new_data.items() }
        # new_data={k:(torch.cat(v) if k not in ['label'] else torch.tensor(v)) for k,v  in new_data.items()}
        new_data['num_faces_per_solid']=face_num.long()
        # new_data['label']=torch.flatten(new_data['label'])

        # assert len(new_data['label'])==len(new_data['face'])

        return new_data

    def __getitem__(self, index):
        sample = self.load_one_sample(self.data[index])
        sample = self.normalize(sample)
        sample = self.padding(sample,max_facet_len=800,padding_mode='circular')
        sample = self.convert_to_float32(sample)
        # sample['face']=torch.rand_like(sample['face'])
        # sample['face_vis_mask']=torch.zeros_like(sample['face_vis_mask'])

        return sample

    def load_samples(self, items, center_and_scale=False):
        '''
            item: {'topo_path':<file_path>,'label':<label>,'face_path':<file_path>,'edge_path':<file_path>}
        '''
        self.data = []
        self.labels = []
        for item in tqdm(items):
            if self.load_label_from_file:
                sample= self.load_topo(item['topo'],item['label'])
            else:
                sample = self.load_topo(item['topo'])
            # sample = self.load_topo(item['topo'])
            # sample=True
            if sample is not None:
                self.data.append(item)
        # logger.debug(self.data[0].keys())

class BRTDataset_cls_online(BRTDataset):
    def set_masking_rate_v2(self,masking_rate):
        self.masking_rate_v2=masking_rate
    def load_topo(self, file_path,masking_rate=None):
        solid = torch.load(file_path)

        edge_index=solid['edge_index']
        wire_index=solid['wire_index']
        adj_face_index=solid['adj_face_index']

        max_face_index_length=10
        max_wire_index_length=10
        max_edge_index_length=10
        max_total_face_length=90

        edge_index_length=[]

        face_num=len(wire_index)
        # if self.masking_rate_v2 is None:
        #     if len(wire_index)>max_total_face_length:
        #         # raise RuntimeError('len(edge_index)>max_total_face_length')
        #         logger.warning('len(edge_index)>max_total_face_length')
        #         return None
        if len(wire_index)>max_total_face_length:
            # raise RuntimeError('len(edge_index)>max_total_face_length')
            logger.warning('len(wire_index)>max_total_face_length')
            return None

        if masking_rate is not None:
            # reserved_num=int(max_total_face_length*masking_rate)
            perm_index=torch.randperm(len(wire_index),dtype=torch.long)
            perm_index_padding=torch.arange(max_total_face_length,dtype=torch.long)
            perm_index_padding[:len(wire_index)]=perm_index
            perm_index=perm_index_padding
        else:
            perm_index=None
            reserved_num=0

        perm_index2=None
        reserved_num=None
        masking_rate_v2_enabled=self.masking_rate_v2 is not None
        if self.masking_rate_v2 is not None:
            reserved_num=int(max_total_face_length*self.masking_rate_v2)
            if reserved_num<len(wire_index):
                perm_index2=torch.randperm(len(wire_index),dtype=torch.long,device='cpu')
                # perm_index2_padding=torch.arange(max_total_face_length,dtype=torch.long)
                # perm_index2_padding[:len(wire_index)]=perm_index
                # perm_index=perm_index_padding
                reverse_map=torch.scatter(torch.zeros(len(wire_index),dtype=torch.long),0,perm_index2,torch.arange(len(wire_index),dtype=torch.long)).to(device='cpu')
            else:
                masking_rate_v2_enabled=False

        if masking_rate_v2_enabled:
            # print(len(wire_index),len(perm_index2),len(reverse_map),len(adj_face_index),reserved_num,)
            adj_face_index=[adj_face_index[perm_index2[idx]] for idx in range(reserved_num)]
            wire_index=[wire_index[perm_index2[idx]] for idx in range(reserved_num)]

        wire_index_length=torch.zeros(len(wire_index),dtype=torch.long)
        adj_face_index_length=torch.zeros(len(adj_face_index),dtype=torch.long)

        for fid,index_lst in enumerate(adj_face_index):
            if masking_rate_v2_enabled:
                index_lst=[reverse_map[i] for i in index_lst if  i<face_num and reverse_map[i]<reserved_num]
            if len(index_lst)>max_face_index_length:
                # logger.warning('adj_face_index_length[fid]>max_face_index_length')
                # return None
                index_lst=random.sample(index_lst,max_face_index_length)
            adj_face_index_length[fid]=len(index_lst)
            while len(index_lst)<max_face_index_length:
                index_lst.append(0)
            adj_face_index[fid]=index_lst

        # wire_index is a list, each item in the list is the wire indices of one face
        for fid,index_lst in enumerate(wire_index):
            if masking_rate_v2_enabled:
                index_lst=[reverse_map[i] for i in index_lst if i<face_num and reverse_map[i]<reserved_num]
            if len(index_lst)>max_wire_index_length:
                # logger.warning('wire_index_length[fid]>max_wire_index_length')
                # return None
                index_lst=random.sample(index_lst,max_wire_index_length)
            wire_index_length[fid]=len(index_lst)
            while len(index_lst)<max_wire_index_length:
                index_lst.append(0)
            wire_index[fid]=index_lst

        # edge_index is a list, each item in the list is the edge indices of one wire
        for wid,index_lst in enumerate(edge_index):
            if len(index_lst)>max_edge_index_length:
                # logger.warning('edge_index_length[fid]>max_edge_index_length')
                # return None
                index_lst=random.sample(index_lst,max_edge_index_length)
                edge_index[wid]=index_lst # is this In-Place Modification safe? Answer: Yes
            edge_index_length.append(len(index_lst))
            while len(index_lst)<max_edge_index_length:
                index_lst.append(0)

        edge_index_length=torch.tensor(edge_index_length,dtype=torch.long)
        edge_index=torch.tensor(edge_index,dtype=torch.long)
        if type(adj_face_index)==list:
            adj_face_index=torch.tensor(adj_face_index,dtype=torch.long)
        if type(wire_index)==list:
            wire_index=torch.tensor(wire_index,dtype=torch.long)

        return {
            'edge':solid['edge'],
            'edge_index_length':edge_index_length,'wire_index_length':wire_index_length,
                'adj_face_index_length':adj_face_index_length,
                'edge_index':edge_index,'wire_index':wire_index,'adj_face_index':adj_face_index},perm_index,perm_index2,reserved_num
    def load_one_sample(self, item):
        # Load the graph using base class method
        """
        Returns:
            sample: token_update arrays
            label: [{'instantces':[],'embedding':<txt_embedding>} for each feature]
        """
        face = self.load_face(item['face'])
        # edge = self.load_edge(item['edge_path'])
        topo = self.load_topo(item['topo'],self.masking_rate)


        if face is None or topo is None:
            # logger.debug('None face or topo'+','+str(face is None)+','+str(topo is None))
            return None
        data={}
        topo,perm_index,perm_index_v2,reserved_num=topo

        # torch._assert(len(face['face'])==len(topo['wire_index']),f"{len(face['face'])},{len(topo['wire_index'])},{item['face']},{item['topo']}")

        if self.masking_rate is not None:
            data['perm_index']=perm_index
        # data['reserved_num']=reserved_num

        if self.masking_rate_v2 is not None and perm_index_v2 is not None:
            face['face']=[face['face'][perm_index_v2[idx]] for idx in range(reserved_num)]
            face['face_vis_mask']=[face['face_vis_mask'][perm_index_v2[idx]] for idx in range(reserved_num)]

        data.update(face)
        data.update(topo)
        data['filename']=item['face']
        data['label']=item['label']

        return data

    def __getitem__(self, index):
        sample = self.load_one_sample(self.data[index])
        sample = self.normalize(sample)
        sample = self.padding(sample,padding_mode='circular',max_facet_len=100)
        sample = self.convert_to_float32(sample)
        # sample['face']=torch.rand_like(sample['face'])
        # sample['face_vis_mask']=torch.zeros_like(sample['face_vis_mask'])

        return sample

    def load_samples(self, items, center_and_scale=False):
        '''
            item: {'topo_path':<file_path>,'label':<label>,'face_path':<file_path>,'edge_path':<file_path>}
        '''
        self.data = []
        self.labels = []
        for item in tqdm(items):
            # sample = self.load_one_sample(item)
            sample = self.load_topo(item['topo'])
            if sample is not None:
                self.data.append(item)
        # logger.debug(self.data[0].keys())
