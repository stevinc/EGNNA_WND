import pandas as pd
import torch
import torch.nn as nn
import math
from haversine import haversine
from collections import Counter
from datetime import datetime, timedelta
import argparse


def nearest(items, pivot):
    if min(abs(datetime.strptime(x.parts[3], '%Y_%m_%d') - pivot) for x in items).days > 60:
        return None
    else:
        return min(items, key=lambda x: abs(datetime.strptime(x.parts[3], '%Y_%m_%d') - pivot))


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


def find_topk_neighbours(pickle_file, pickle_file_to_save, excel_file_to_save, topk, Years):
    # open the file and delete the empty rows for the various band
    data_info = pd.read_pickle(pickle_file)
    data_info = data_info[data_info['LST'].map(lambda d: len(d)) > 0]
    data_info = data_info[data_info['SSM_all'].map(lambda d: len(d)) > 0]
    data_info = data_info[(data_info['S2'].map(lambda d: len(d)) > 0) | (data_info['L8'].map(lambda d: len(d)) > 0)]
    # split the dataframe into each single year
    data_info_splitted = [data_info[data_info['Year'] == Years[0]].copy(), data_info[data_info['Year'] == Years[1]].copy(),
                          data_info[data_info['Year'] == Years[2]].copy()]
    data_info_splitted[0].reset_index(drop=True)
    data_info_splitted[1].reset_index(drop=True)
    data_info_splitted[2].reset_index(drop=True)
    # repeat for each Year
    for idx,y in enumerate(Years):
        # CoordXY (Lat-Long) of each point, they are inverted, so flip!!
        CoordXY = [(j, i) for i, j in zip(data_info_splitted[idx]['CoordX'], data_info_splitted[idx]['CoordY'])]
        # calculate the haversine distance between each points
        distances = [[haversine_fn(a, b) for id_a, a in enumerate(CoordXY)] for id_b, b in enumerate(CoordXY)]
        distances = list(map(torch.stack, distances))
        distances = torch.stack(distances)
        # set the distances between each point with itself to a huge number so it will not appear in the neighbourhood
        for d in range(distances.shape[0]):
            distances[d, d] = 10000
        dist_neighbours, indices_neighbours = distances.topk(k=topk, largest=False, dim=1)
        # select the IdPoint corresponding to each neighbour
        IdPoint = data_info_splitted[idx]['IdPoint'].tolist()
        neighbours_col = [[IdPoint[n1] for n1 in n] for n in indices_neighbours]
        # add the column to the dataframe
        data_info_splitted[idx][f'neighbours_best10'] = neighbours_col
    # create the new dataframe and save it
    data_info = pd.concat(data_info_splitted, ignore_index=True)
    data_info.to_pickle(pickle_file_to_save)
    data_info.to_excel(excel_file_to_save, index=False)
    print("End searching for topk neighbours..")


def check_number_of_neighbours(pickle_file_to_save):
    df = pd.read_pickle(pickle_file_to_save)
    df_neigh = df['neighbours'].tolist()
    count = Counter([len(n) for n in df_neigh])
    print(f"Neighbours: {count}")


def create_unrolled_dataset(pickle_file, pickle_file_to_save, excel_file_to_save, delete_neighbours=0):
    print("start creation of unrolled dataset..")
    df = pd.read_pickle(pickle_file)
    # delete row with empty list values in LST
    df = df[df['LST'].map(lambda d: len(d)) > 0]
    # non so farlo tutto in una volta
    df = df[df['SSM_all'].map(lambda d: len(d)) > 0]
    df = df[(df['S2'].map(lambda d: len(d)) > 0) | (df['L8'].map(lambda d: len(d)) > 0)]
    # new column
    IdPoint = df.loc[:, 'IdPoint'].tolist()
    status = df.loc[:, 'Status'].tolist()
    DateInfection = df.loc[:, 'DateInfection'].tolist()
    CoordXY = [(i, j) for i, j in zip(df['CoordX'], df['CoordY'])]
    years = df.loc[:, 'Year'].tolist()
    imgs_list_s2 = df.loc[:, 'S2'].tolist()
    imgs_list_l8 = df.loc[:, 'L8'].tolist()
    imgs_list = [i + j for i, j in zip(imgs_list_s2, imgs_list_l8)]
    neighbours = df.loc[:, 'neighbours_best10'].tolist()
    CoordAccuracy = df.loc[:, 'CoordAccuracy'].tolist()
    data_len = len(imgs_list)
    print("Dataset len: ", data_len)
    # CREATE THE NEW DATAFRAME
    df_new = pd.DataFrame(columns=['Year', 'IdPoint', 'DateInfection', 'CoordXY', 'Status', 'Imgs',
                                   'neighbours_best10', 'CoordAccuracy'])
    total_index = 0
    empty_neighbours = 0
    for l in range(data_len):
        print(l)
        neighbours_indices = [IdPoint.index(n) for n in neighbours[l]]
        images_dates_neighbours = []
        for n in neighbours_indices:
            images_dates_neighbours.append([im for im in imgs_list[n]])
        for idx, el in enumerate(imgs_list[l]):
            # find the nearest image for each neighbour
            date_el = datetime.strptime(el.parts[3], '%Y_%m_%d')
            neighbours_imgs = []
            for n in range(len(neighbours_indices)):
                near_img = nearest(images_dates_neighbours[n], date_el)
                if near_img:
                    neighbours_imgs.append(near_img) # .strftime('%Y_%m_%d'))
            if not neighbours_imgs:
                empty_neighbours += 1
            df_new.loc[total_index+idx] = [years[l], IdPoint[l], DateInfection[l], CoordXY[l], status[l], el,
                                           neighbours_imgs, CoordAccuracy[l]]
        total_index += len(imgs_list[l])
    df_new.to_pickle(pickle_file_to_save)
    df_new.to_excel(excel_file_to_save, index=False)
    print(f"image with no neighbours{empty_neighbours}")
    print("End new dataset creation..check!")


if __name__ == '__main__':
    pickle_file = "WND_171819_complete.pkl"
    Years = [2017, 2018, 2019]
    topk = 10

    # pickle_file_to_save = f"WND_171819_complete_neighbours_topk{topk}_haversine_correct.pkl"
    # excel_file_to_save = f"WND_171819_complete_neighbours_topk{topk}_haversine_correct.xlsx"
    # find_topk_neighbours(pickle_file, pickle_file_to_save, excel_file_to_save, topk, Years)

    #
    # check_number_of_neighbours(pickle_file_to_save)
    # CREATE UNROLLED VERSION OF THE DATASET
    # FILE TO SAVE
    pickle_file_to_save = f"WND_171819_complete_neighbours_topk{topk}_irregular_unrolled_haversine_correct_ensemble_teramo.pkl"
    excel_file_to_save = f"WND_171819_complete_neighbours_topk{topk}_irregular_unrolled_haversine_correct_ensemble_teramo.xlsx"
    create_unrolled_dataset(f"WND_171819_complete_neighbours_topk{topk}_haversine_correct.pkl", pickle_file_to_save, excel_file_to_save)