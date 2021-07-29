import pandas as pd
from os.path import join
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import ntpath
from multiprocessing import Process

def calculate_label_dist(sampling_dir):
    print(sampling_dir)
    label_dist = defaultdict(lambda: 0)
    for fn in tqdm(glob(join(sampling_dir,'*Meter.csv'))):
        df = pd.read_csv(fn,usecols=['Label'], dtype={'Label':str})
        for key, value in df['Label'].value_counts().iteritems():
            label_dist[key]+=value

    with open(join(sampling_dir,'label_dist.csv'),'w') as f:
        for key in sorted(label_dist.keys()):
            f.write('{},{}\n'.format(key,label_dist[key])) 

dataroot = '/data/juma/data/ids18/CSVs_r_0.001_m_1.0/SR_10.0/'
sampling_dirs = [d for d in glob(join(dataroot,'*_l'))]

procs = [ Process(target=calculate_label_dist, args=[sdir]) for sdir in sampling_dirs]
for p in procs: p.start()
for p in procs: p.join()


