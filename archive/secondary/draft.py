#in pre processing stage we will split (and write) the data with non-overlapping flowids

import numpy as np
from os.path import join as os_join
import os
import csv
from numpy import genfromtxt

dataroot = '/media/bek/8E92899D92898A83/arash/CIC-IDS-2017/PCAPs/sr10/'
#dataroot = '/media/bek/8E92899D92898A83/arash/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabelling'

SAMPLING_RATE = 100


def makedir(filenames):
        for filename in filenames:
                    directory = os.path.dirname(os.path.abspath(filename))
                    if not os.path.exists(directory):
                        os.makedirs(directory)


def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]


def read_csv(filename, sampling_rate=100):
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            print ('len: {} | line[{}] = {}'.format(len(line),i, line))
            if i==3:
                return


filenames = get_filenames(dataroot)

#N is the number of rows
for filename in filenames:
    print(filename)
    data = read_csv(os_join(dataroot,filename), sampling_rate=SAMPLING_RATE)
