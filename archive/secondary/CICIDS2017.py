import numpy as np
import os
import csv
from os.path import join as os_join
from numpy import genfromtxt


def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name)) and not name.startswith(".~lock.") and name.endswith(".csv") and not name.startswith("distribution")]


def read_csv_header(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        return header    


def read_csv(filename):
    with open(filename,encoding='utf-8') as csv_file:
        print('Reading {} ...'.format(filename))
        data= genfromtxt(filename, delimiter=',',dtype=np.str_)
        data = data[1:,:] # remove header
        if data.shape[1]==85:
            data = np.delete(data,-24,axis=1)
        data = np.char.strip(data,'"')
        print("data read:",data.shape)
    return data


def read_data(dataroot):
    filenames = get_filenames(dataroot)
    data = np.empty((0,84))
    for filename in filenames:
        data_part = read_csv(os_join(dataroot,filename))
        print(data.shape,data_part.shape)
        data=np.concatenate((data,data_part),axis=0)
    return data

