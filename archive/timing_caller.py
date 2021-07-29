import os
import glob
from multiprocessing import Pool
from classify_for_timing import classify


root = '/home/juma/data/net_intrusion/CIC-IDS-2018/CSVs_r_0.001/SR_10/'
#root = '/mnt/sda_dir/juma/data/net_intrusion/CIC-IDS-2018/CSVs_r_0.2/SR_10/'
#root = '/media/juma/data/research/intrusion_detection/dataset/CIC-IDS-2018/CSVs/SR_10/'
#root = '/home/juma/data/net_intrusion/CIC-IDS-2018/cache_mem_limit_archive/CSVs_mem_100/SR_1'
#file_ending = '*.pcap_Flow.csv'
file_ending =  '*_CICFlowMeter.csv'


def get_immediate_subdirs(a_dir, only=''):
    if only=='':
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir) 
                if os.path.isdir(os.path.join(a_dir, name)) and name.endswith('_l') and 'whole' not in name]
    else:
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name)) and name.endswith('_l') and (only in name )]


def execute(cmd):
	print(cmd)
	dataroot,classifier_name,file_ending = cmd
	classify(dataroot,classifier_name)


#classifier_name = 'tree'
#classifier_name = 'forest'
#classifier_name = 'softmax'
classifier_name='cnn'

args = []
dirs = get_immediate_subdirs(root)
for d in dirs:
    cmd = (d,classifier_name,file_ending)
    args.append(cmd)
args = sorted(args)


#multi proc
#p = Pool(processes = 6)
#p.map(execute,args)
for i,arg in enumerate(args):
    print(i)
    execute(arg)

