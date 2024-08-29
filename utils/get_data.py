import cv2
from matplotlib import pyplot as plt
import albumentations as A
import numpy as np
import scipy.sparse as sp
import torch
import random
from utils.params import args
import pandas as pd

# An example for text node extraction.


class NodeTextDataset():
    def __init__(self, entity_id,label):

        # self.text = list(text)

        # self.encoded_text = tokenizer(
        #     self.text, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt'
        # )

        self.entity_id = entity_id
        self.label = label

    def __getitem__(self, idx):
        # item = {
        #     key: torch.tensor(values[idx])
        #     for key, values in self.encoded_text.items()
        # }
        # # item['text'] = self.text[idx]
        # item['entity'] = self.entity_id[idx]
        # item['label'] = self.label[idx]
        item = {
        'entity': self.entity_id[idx],
        'label': self.label[idx]
        }

        return item


    def __len__(self):
        return len(self.entity_id)

# class NodeMultimodalDataset():
#     def __init__


# get transformation for image data
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(args.image_size, args.image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(args.image_size, args.image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

# def split_imbalance(labels,train_ratio,val_ratio,test_ratio,imbalance_ratio):

#     num_classes = len(set(labels.tolist()))
#     c_idxs = [] # class-wise index
#     train_idx = []
#     val_idx = []
#     test_idx = []
#     c_num_mat = np.zeros((num_classes,3)).astype(int)
#     label_max = int(max(labels.tolist())+1)
#     minority_index = [item for item in range(label_max) if labels.tolist().count(item) <  len(labels.tolist())/num_classes]

#     for i in range(num_classes):
#         c_idx = (labels==i).nonzero()[:,-1].tolist()
#         c_num = len(c_idx)
#         if num_classes > 2:
#             if i in minority_index: c_num = int(len(labels.tolist()) / num_classes * imbalance_ratio)
#         else:
#             if i in minority_index: c_num = int((len(labels.tolist())- labels.tolist().count(i)) * imbalance_ratio)

#         print('The number of class {:d}: {:d}'.format(i,c_num))
#         random.shuffle(c_idx)
#         c_idxs.append(c_idx)

#         if c_num <4:
#             if c_num < 3:
#                 print("too small class type")
#             c_num_mat[i,0] = 1
#             c_num_mat[i,1] = 1
#             c_num_mat[i,2] = 1
#         else:
#             c_num_mat[i,0] = int(c_num/10 *train_ratio)
#             c_num_mat[i,1] = int(c_num/10 * val_ratio)
#             c_num_mat[i,2] = int(c_num/10 * test_ratio)


#         train_idx = train_idx + c_idx[:c_num_mat[i,0]]

#         val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
#         test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

#     random.shuffle(train_idx)
#     # print(c_num_mat)
#     print(train_idx)
#     print(val_idx)


#     return train_idx, val_idx, test_idx, c_num_mat

def split_imbalance(labels,train_ratio,val_ratio,test_ratio,imbalance_ratio):

    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    label_max = int(max(labels.tolist())+1)
    minority_index = [item for item in range(label_max) if labels.tolist().count(item) <  len(labels.tolist())/num_classes]

    # idx_label = args.id_label_path
    # df = pd.read_csv(idx_label)
    # train_idx = df['id'].values[:(3624-717)]
    # val_idx = df['id'].values[(3624-717):]
    # test_idx = df['id'].values[(3624-717):]
    
    
    

    

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        if num_classes > 2:
            if i in minority_index: c_num = int(len(labels.tolist()) / num_classes * imbalance_ratio)
        else:
            if i in minority_index: c_num = int((len(labels.tolist())- labels.tolist().count(i)) * imbalance_ratio)

        print('The number of class {:d}: {:d}'.format(i,c_num))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/10 *train_ratio)
            c_num_mat[i,1] = int(c_num/10 * val_ratio)
            c_num_mat[i,2] = int(c_num/10 * test_ratio)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)
    print(c_num_mat)
    print(train_idx)
    print(val_idx)


    return train_idx, val_idx, test_idx, c_num_mat

def split_balance(labels,train_ratio,val_ratio,test_ratio):

    num_classes = len(set(labels.tolist()))
    c_idxs = []
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    count_0, count_1 = labels.tolist().count(0), labels.tolist().count(1)
    count_min = min(count_0,count_1)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = count_min

        print('The number of class {:d}: {:d}'.format(i,c_num))

        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/10 *train_ratio)
            c_num_mat[i,1] = int(c_num/10 * val_ratio)
            c_num_mat[i,2] = int(c_num/10 * test_ratio)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]


    return train_idx, val_idx, test_idx, c_num_mat

def split_genuine(labels,train_ratio,val_ratio,test_ratio):

    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('The number of class {:d}: {:d}'.format(i,c_num))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/10 *train_ratio)
            c_num_mat[i,1] = int(c_num/10 * val_ratio)
            c_num_mat[i,2] = int(c_num/10 * test_ratio)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]


    return train_idx, val_idx, test_idx, c_num_mat

def load_data_for_pretrain():

    # parser.add_argument('--multimedia_feature_path', type=str,default=r'./data/fakesv_data/node_features.npy',help='path of aminer feature')
    # parser.add_argument('--text_feature_path', type=str,default=r'./data/fakesv_data/text_features.npy',help='path of aminer feature')
    # parser.add_argument('--ocr_feature_path', type=str,default=r'./data/fakesv_data/ocr_features.npy',help='path of aminer feature')
    # parser.add_argument('--image_feature_path', type=str,default=r'./data/fakesv_data/image_features.npy',help='path of aminer feature')
    # parser.add_argument('--audio_feature_path', type=str,default=r'./data/fakesv_data/audio_features.npy',help='path of aminer feature')
    # parser.add_argument('--video_feature_path', type=str,default=r'./data/fakesv_data/video_features.npy',help='path of aminer feature')
    # parser.add_argument('--comment_feature_path', type=str,default=r'./data/fakesv_data/comment_features.npy',help='path of aminer feature')
    # parser.add_argument('--id_label_path', type=str, default=r'./data/fakesv_data/id_label.csv',help='to get the finetune features')
    
    embed_dir = args.multimedia_feature_path
    text_embed = args.text_feature_path
    ocr_embed = args.ocr_feature_path
    image_embed = args.image_feature_path
    audio_embed = args.audio_feature_path
    video_embed = args.video_feature_path
    comment_embed = args.comment_feature_path
    idx_label = args.id_label_path
    
    relation_dir = args.relation_path
    pos_dir = args.pos_path

    features = np.load(embed_dir)
    text_features = np.load(text_embed)
    ocr_features = np.load(ocr_embed)
    img_features = np.load(image_embed)
    audio_features = np.load(audio_embed)
    video_features = np.load(video_embed)
    comment_features = np.load(comment_embed)
    # idx_features_labels = np.genfromtxt(embed_dir,
    #                                     dtype=np.dtype(str), delimiter=',', invalid_raise=True)
    
    features = sp.csr_matrix(features, dtype=np.float32)
    idx_label = pd.read_csv(idx_label)
    label_data = idx_label['label'].values
    idx = np.array(idx_label['id'].values, dtype=np.int32)

    # label_data = idx_features_labels[:,0]
    label = encode_onehot(label_data)

    # build graph
    # idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    # print(idx_map)

    total_num = features.shape[0]
    

    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)
    # print(edges_unordered)
    # print(edges_unordered)
    # print(idx_map.get)
    # print(edges_unordered.flatten())
    # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  dtype=np.int32).reshape(edges_unordered.shape)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # print(edges)
    edges = edges_unordered
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(total_num,total_num),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    number = 3624
    pos_unordered = np.genfromtxt(pos_dir,
                                    dtype=np.int32)
    #####################
    # pos_unordered = pos_unordered < pos_unordered
    # print(pos_unordered)
    pos = np.array(list(map(idx_map.get, pos_unordered.flatten())),
                     dtype=np.int32).reshape(pos_unordered.shape)
    pos = sp.coo_matrix((np.ones(pos.shape[0]), (pos[:, 0], pos[:, 1])),
                        shape=(number,number),
                        dtype=np.int32)
    # build symmetric adjacency matrix
    pos = pos + pos.T.multiply(pos.T > pos) - pos.multiply(pos.T > pos)
    pos = pos + sp.eye(pos.shape[0])

    # adj = adj.multiply(pos.T*0.5)
    ############
    # pos = pos < pos
    labels = torch.LongTensor(np.where(label)[1]).to(args.device)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(args.device)
    pos = sparse_mx_to_torch_sparse_tensor(pos).to(args.device)
    features = {'feature': torch.FloatTensor(np.array(features.todense())).to(args.device)}

    return adj, features, labels, pos
