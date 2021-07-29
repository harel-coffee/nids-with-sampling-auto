#in pre processing stage we will split (and write) the data with non-overlapping flowids

import numpy as np
from CICIDS2017 import read_csv
from os.path import join as os_join
import os
import csv

dataroot = '/home/jumabek/IntrusionDetectionSampling/data/GeneratedLabelledFlows/TrafficLabelling'


SAMPLING_RATE = 100


def makedir(filenames):
        for filename in filenames:
                    directory = os.path.dirname(os.path.abspath(filename))
                    if not os.path.exists(directory):
                        os.makedirs(directory)


def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]


def write_csv(data, filename):
    with open( filename,"w") as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for row in data:
            wr.writerow(row)


def write_data(data, input_file):
    output_file = input_file.replace("TrafficLabelling", "TrafficLabelling_NS_SR{}".format(SAMPLING_RATE)) 
    makedir([output_file])
    write_csv(data,output_file)
    


filenames = get_filenames(dataroot)

#N is the number of rows
for filename in filenames:
    data = read_csv(os_join(dataroot,filename), sampling_rate=SAMPLING_RATE)
    write_data(data, os_join(dataroot,filename))
