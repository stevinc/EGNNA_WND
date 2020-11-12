from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import SequentialSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import math
from sinkhorn_knopp import sinkhorn_knopp as skp
from graph.block_diag_matrix import block_diag


def haversine_fn(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    distances = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return torch.tensor(distances*0.001)


def double_stochastic_norm_sinkhorn(similarity):
    # guardare il loro codice
    sk = skp.SinkhornKnopp()
    DS = sk.fit(similarity)
    return torch.from_numpy(DS).float()


def compute_adj_matrix_coord(CoordXY):
    """
    Compute adjacency matrix from a tensor of n images.
    """
    s = 1.
    # compute the distances between all aux data in the current batch
    distances = [[haversine_fn(a, b) for a in CoordXY] for b in CoordXY]
    distances = list(map(torch.stack, distances))
    distances = torch.stack(distances)
    similarity = torch.exp(- (distances) / (2. * s ** 2))
    adj_matrix = double_stochastic_norm_sinkhorn(similarity)
    return adj_matrix


def compute_adj_matrix_aux(aux_bands):
    """
    Compute adjacency matrix from a tensor of n images.
    """
    s = 1.
    l1 = nn.L1Loss()
    # l2 = nn.MSELoss()
    # compute the distances between all aux data in the current batch
    distances = [[l1(a, b) for a in aux_bands] for b in aux_bands]
    distances = list(map(torch.stack, distances))
    distances = torch.stack(distances)
    # compute gaussian kernel in order to obtain a similarity matrix
    adj_matrix = torch.exp(- (distances) / (2. * s ** 2))
    # normalization choice
    # adj_matrix = double_stochastic_norm_sinkhorn(similarity)
    return adj_matrix

# function for data augmentation
def custom_augmentation(image):
    rnd = np.random.random_sample()
    if 0.25 <= rnd <= 0.50:
        image = TF.to_pil_image(image)
        image = TF.vflip(image)
    elif 0.50 < rnd <= 0.75:
        image = TF.to_pil_image(image)
        image = TF.hflip(image)
    else:
        return image
        image = TF.to_tensor(image)
    return image


def id_collate(batch):
    # concatenate the neighbour images on the batch dimension
    imgs_neighbours = torch.cat([list(_batch)[0] for _batch in batch], dim=0)
    labels = torch.tensor([_batch[1] for _batch in batch]).view(-1)
    points = [_batch[4] for _batch in batch]
    date = [_batch[5] for _batch in batch]
    date_infection = [_batch[6] for _batch in batch]
    coordAcc = [_batch[7] for _batch in batch]
    if len(batch[0][2].size()) == 3:
        adj_matrix = torch.stack([_batch[2] for _batch in batch], dim=1)
        blocks_adj_matrix = [block_diag(adj)for adj in adj_matrix]
        blocks_adj_matrix = torch.stack(blocks_adj_matrix, 0)
    else:
        adj_matrix = torch.stack([_batch[2] for _batch in batch], dim=0)
        blocks_adj_matrix = block_diag(adj_matrix)
    return imgs_neighbours, labels, blocks_adj_matrix, batch[0][3], points, date, date_infection, coordAcc


class EsaDatasetGraph(Dataset):
    def __init__(self, dataset_file, nas_path="/nas/softechict-nas-2/svincenzi/store22.eo.esa.int/",
                 bands=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dsize=224, RGB=0, mode='train', val=0,
                 neighbours_labels=0, adjacency_type='ones'):
        # saving the current dir of this subfolder for local imports
        self.curr_dir = Path(__file__).parent
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Dataset
        self.data_info = pd.read_pickle(dataset_file)
        if mode == 'train' and val == 0:
            self.data_info = self.data_info[(self.data_info['Year'] == 2017) | (self.data_info['Year'] == 2018)]
        elif mode == 'train' and val == 1:
            self.data_info = self.data_info[(self.data_info['Year'] == 2018)]
        elif mode == 'val':
            self.data_info = self.data_info[(self.data_info['Year'] == 2017)]
        elif mode == 'test':
            self.data_info = self.data_info[self.data_info['Year'] == 2019]
        else:
            raise ValueError("wrong inputs, select mode=train,test or val; and val=0 or 1...")
        self.data_info = self.data_info.reset_index(drop=True)
        self.IdPoint = self.data_info.loc[:, 'IdPoint'].tolist()
        self.CoordXY = self.data_info.loc[:, 'CoordXY'].tolist()
        self.labels = self.data_info.loc[:, 'Status'].tolist()
        self.path_imgs = self.data_info.loc[:, 'Imgs'].tolist()
        self.neighbours_path = self.data_info.loc[:, 'neighbours_best10'].tolist()
        self.infection_date = self.data_info.loc[:, 'DateInfection'].tolist()
        self.CoordAcc = self.data_info.loc[:, 'CoordAccuracy'].tolist()
        # Calculate len
        self.data_len = len(self.path_imgs)
        print("Dataset len: ", self.data_len)
        # bands
        self.bands = torch.BoolTensor(bands)
        # nas path
        self.nas_path = nas_path
        # image size
        self.dsize = dsize
        # use only RGB
        self.RGB = RGB
        # flag for using the labels of the neighbours, if True classify all the images in the neighbourhood
        self.neighbours_labels = neighbours_labels
        # select the type of adjacency
        self.adjacency_type = adjacency_type

    def load_tensor_image(self, imgs_file: str) -> torch.Tensor:
        spectral_img = torch.load(imgs_file + '/spectral.pt')
        spectral_img = torch.squeeze(
            nn.functional.interpolate(input=torch.unsqueeze(spectral_img, dim=0), size=self.dsize))
        # choose the bands to keep
        spectral_img = spectral_img[self.bands]
        if self.RGB:
            if spectral_img.shape[0] == 3:
                spectral_img = torch.flip(spectral_img, [0])
            else:
                rgb = torch.flip(spectral_img[:3], [0])
                spectral_img = torch.cat((rgb, spectral_img[3:]), 0)
        return spectral_img

    def __getitem__(self, index):
        # obtain the right folder
        imgs_file = self.nas_path + str(self.path_imgs[index])
        spectral_img = self.load_tensor_image(imgs_file=imgs_file)
        # CoordXY
        CoordXY_neighbours = [torch.tensor(self.CoordXY[index])]
        # eventually augmentation, not now
        imgs_neighbours = [spectral_img]
        neighbours_labels = [self.labels[index]]
        # take the neighbours of the current images
        for n in self.neighbours_path[index]:
            imgs_neighbours.append(self.load_tensor_image(self.nas_path + str(n)))
            # find the neighbour index to obtain its CoordXY and labels
            index_n = self.IdPoint.index(n.parts[2][6:])
            CoordXY_neighbours.append(torch.tensor(self.CoordXY[index_n]))
            neighbours_labels.append(self.labels[index_n])
        imgs_neighbours = torch.stack(imgs_neighbours, dim=0)
        CoordXY_neighbours = torch.stack(CoordXY_neighbours, dim=0)
        if self.adjacency_type == 'distance':
            adj_matrix = compute_adj_matrix_coord(CoordXY_neighbours)
        elif self.adjacency_type == 'ones':
            adj_matrix = torch.ones(6, 6)
        elif self.adjacency_type == 'lst':
            adj_matrix = compute_adj_matrix_aux(imgs_neighbours[:, -1].mean(dim=[1, 2]))
            imgs_neighbours = imgs_neighbours[:, :-1]
        elif self.adjacency_type == 'ssm':
            adj_matrix = compute_adj_matrix_aux(imgs_neighbours[:, -1].mean(dim=[1, 2]))
            imgs_neighbours = imgs_neighbours[:, :-1]
        elif self.adjacency_type == 'lst-ssm-separate':
            adj_matrix_lst = compute_adj_matrix_aux(imgs_neighbours[:, -2].mean(dim=[1, 2]))
            adj_matrix_ssm = compute_adj_matrix_aux(imgs_neighbours[:, -1].mean(dim=[1, 2]))
            adj_matrix = torch.stack((adj_matrix_lst, adj_matrix_ssm), 0)
            imgs_neighbours = imgs_neighbours[:, :-2]
        elif self.adjacency_type == 'lst-srtm-ssm-separate':
            adj_matrix_lst = compute_adj_matrix_aux(imgs_neighbours[:, -3].mean(dim=[1, 2]))
            adj_matrix_srtm = compute_adj_matrix_aux(imgs_neighbours[:, -2].mean(dim=[1, 2]))
            adj_matrix_ssm = compute_adj_matrix_aux(imgs_neighbours[:, -1].mean(dim=[1, 2]))
            adj_matrix = torch.stack((adj_matrix_lst, adj_matrix_srtm, adj_matrix_ssm), 0)
            imgs_neighbours = imgs_neighbours[:, :-3]
        if self.neighbours_labels:
            return imgs_neighbours, neighbours_labels, adj_matrix, imgs_neighbours.shape[0], \
                   self.path_imgs[index].parts[2], self.path_imgs[index].parts[3], self.infection_date[index], self.CoordAcc[index]
        else:
            return imgs_neighbours, self.labels[index], adj_matrix, imgs_neighbours.shape[0], \
                   self.path_imgs[index].parts[2], self.path_imgs[index].parts[3], self.infection_date[index], self.CoordAcc[index]

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    # path to the dataset
    nas_path_dataset = "/nas/softechict-nas-2/svincenzi/store22.eo.esa.int/"
    # bands to use
    bands = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    # dataset definition
    esa_train = EsaDatasetGraph(dataset_file='WND_171819_complete_neighbours_topk10_unrolled_haversine_correct_ensemble_teramo.pkl', nas_path=nas_path_dataset,
                                bands=bands, dsize=224, RGB=1, mode='train', val=0, neighbours_labels=1, adjacency_type = 'lst-srtm-ssm-separate')

    # # sampler
    train_sampler = SequentialSampler(esa_train)

    # # loader definition, set the number of workers and the batch size
    num_workers = 0
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(esa_train, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers, collate_fn=id_collate)

    # loop
    for batch_idx, (in_bands, labels, blocks_adj_matrix, n, points, date, DateInfection, CoordAcc) in enumerate(train_loader):
        print(batch_idx)
    print("end training set")

