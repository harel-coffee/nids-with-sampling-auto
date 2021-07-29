from os.path import join 
import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import pandas as pd
import glob
import time
import ntpath
from tqdm import tqdm
import subprocess

from utils import get_ids18_mappers
from utils import get_cols4eval
from utils import get_dtype, get_dtype4normalized
from utils import encode_label
from classifier_settings import get_args, get_classifier
from classifier_settings import ClassifierLoader
from bookkeepers import result_logger_ids18
import timeit


def inference(clf,x):
    tick = time.time()
    pred = clf.predict(x)
    return time.time()-tick

def predict(df,clf):
        x = df.drop(columns=['Flow ID','Day','Label','Timestamp']).values
        batch_sizes = [16,64,256,1024, 4096]
        inference_times = {}
        for bs in batch_sizes:
            inference_times[bs]= np.array([clf.inference_n_time(x,bs) for i in tqdm(range(100))]).mean()
            #tick = time.time()
            #pred = clf.predict(x_per_record, bs=bs)
            #inference_times[bs] = time.time()-tick
        
        return inference_times



def evaluator(dataroot,classifier_name):
    K=10
   
    print("\nevaling ",dataroot)
    gt_num_class = pd.read_csv(join(dataroot,'{}fold_0.csv'.format(K)),usecols=['Label'])['Label'].nunique()

    # load Classifier
    classifier_args, config = get_args(classifier_name, class_weight=None )
    print("Balancing technique: ", classifier_args['balance'])
    pre_fingerprint = join(dataroot, 'c_{}'.format(classifier_name))
    fingerprint = join(pre_fingerprint + config,'K_{}'.format(K))
    logdir = join(fingerprint,'log')

    gt_classes = None 
    for test_index in range(K):
        print("*************  Testing holdout ", test_index,'************')
        runs_dir = join(logdir,'{}'.format(test_index))
        print(runs_dir)

        print("with model: ",runs_dir)
        classifier_args['runs_dir'] = runs_dir

        loader = ClassifierLoader()
        clf = loader.load(classifier_args)
        # classifier loaded


        # load data
        col_names = get_cols4eval()
        col_names.append('Timestamp')
        test_csv_file = join(dataroot, '{}fold_{}.csv'.format(K,test_index))
        df = pd.read_csv(test_csv_file,usecols=col_names,dtype=get_dtype4normalized(), nrows=10000)#,skiprows=skip_idx)
        x = df.drop(columns=['Flow ID','Label','Timestamp']).values
        # Done    
        for i in range(10):
            inference_time = inference(clf,x)
            print(inference_time)
        break
if __name__=='__main__':
    if 'noWS'=='WS':
        #datadir = '/data/juma/data/ids18/CSVs/WS_l'
        datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs_r_1.0_m_1.0/WS_l'
        evaluator(datadir, 'tree')
    else:
        csv_dirs = ['CSVs_r_1.0_m_1.0', 'CSVs_r_0.1_m_1.0', 'CSVs_r_0.01_m_1.0', 'CSVs_r_0.001_m_1.0']
        sampler_dirs = {}
        sampler_dirs['SI_10'] = ['SFS_SI_9.77_l', 'SGS_e_0.05_l','SRS_SI_10_l','FFS_(8,16,4)_l']
        sampler_dirs['SI_100'] = ['SFS_SI_95.33_l', 'SGS_e_1_l','SRS_SI_100_l','FFS_(8,16,40)_l']
        sampler_dirs['SI_1000'] = ['SFS_SI_685.08_l','SRS_SI_1000_l','SGS_e_11.5_l','FFS_(8,16,400)_l']
        samp_inter = 'SI_100'
        for csv_dir in csv_dirs:
            for d in sampler_dirs[samp_inter]:
                datadir = '/data/juma/data/ids18/{}/{}/{}'.format(csv_dir,samp_inter,d)
                evaluator(datadir,'forest')
            
