import numpy as np
import os
import csv
from os.path import join
from numpy import genfromtxt
import time
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import pickle 
import ntpath


dataroot = '/home/juma/data/net_intrusion/CIC-IDS-2018'

def get_ddos19_root():
    return '/data/juma/data/ddos/'
def get_ids18_root():
    return '/data/juma/data/net_intrusion/ids18/'
    #return '/hdd/juma/data/net_intrusion/ids18'
#dataroot = '/mnt/sda_dir/juma/data/net_intrusion/CIC-IDS-2018/'
#dataroot = '/media/juma/data/research/intrusion_detection/dataset/CIC-IDS-2018/'
def getSeed():
    SEED=234
    return SEED


def get_class_weights(dataroot,p=1):
    df = pd.read_csv(join(dataroot,'label_dist.csv'), names=['Label','Count'])
    label_to_id, id_to_label, _ = get_ids18_mappers()
    #order of labels should be same as label ids on train data
    counts = []
    for i in range(len(id_to_label)):
        label = id_to_label[i]
        if label in df['Label'].values:
            c = df[df['Label']==label]['Count'].iloc[0]
            counts.append(c)
        else:
            print('no {} for class weights'.format(label))
            counts.append(0)

    counts = np.array(counts)
    normed_weights = [1-(count/sum(counts)) for count in counts]
    return np.array(normed_weights)


def ensure_dir(dir_path):
    #print('making dir', dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def ensure_dirs(list_of_dir_path):
    for dir_path in list_of_dir_path:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def encode_label( str_labels):
    label_to_id, id_to_label, _ = get_ids18_mappers()
    return [label_to_id[str_label] for str_label in str_labels]


def make_value2index(attacks):
    #make dictionary
    attacks = sorted(attacks)
    d = {}
    counter=0
    for attack in attacks:
        d[attack] = counter
        counter+=1
    return d


def get_cols4eval():
    return ['Flow ID','Dst Port', 'Protocol', 
       'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
       'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
       'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
       'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
       'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
       'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
       'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
       'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
       'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
       'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
       'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
       'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
       'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
       'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
       'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
       'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
       'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
       'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
       'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
       'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
       'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label'] # Day is removed, because normal flowid is split seperately per day


def get_cols4ml():
    return ['Dst Port', 'Protocol', 
       'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
       'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
       'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
       'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
       'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
       'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
       'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
       'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
       'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
       'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
       'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
       'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
       'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
       'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
       'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
       'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
       'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
       'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
       'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
       'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
       'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']


def get_cols4norm():
    return ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
       'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
       'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
       'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
       'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
       'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
       'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
       'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
       'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
       'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
       'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
       'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
       'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
       'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
       'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
       'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
       'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
       'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
       'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
       'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
       'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']


def get_dtype():
    return {'Flow ID':str , 'Dst Port':int , 'Protocol':int, 'Timestamp': str,
       'Flow Duration':int , 'Tot Fwd Pkts':int , 'Tot Bwd Pkts': int,
       'TotLen Fwd Pkts':int , 'TotLen Bwd Pkts':int , 'Fwd Pkt Len Max':int,
       'Fwd Pkt Len Min':int , 'Fwd Pkt Len Mean':float , 'Fwd Pkt Len Std':float,
       'Bwd Pkt Len Max':int , 'Bwd Pkt Len Min':int , 'Bwd Pkt Len Mean':float,
       'Bwd Pkt Len Std':float , 'Flow Byts/s':float , 'Flow Pkts/s':float, 'Flow IAT Mean':float,
       'Flow IAT Std':float , 'Flow IAT Max':int , 'Flow IAT Min':int , 'Fwd IAT Tot':int,
       'Fwd IAT Mean':float , 'Fwd IAT Std':float , 'Fwd IAT Max':int , 'Fwd IAT Min':int,
       'Bwd IAT Tot':int , 'Bwd IAT Mean':float , 'Bwd IAT Std':float , 'Bwd IAT Max':int,
       'Bwd IAT Min':int , 'Fwd PSH Flags':int , 'Bwd PSH Flags':int , 'Fwd URG Flags':int,
       'Bwd URG Flags':int , 'Fwd Header Len':int , 'Bwd Header Len':int , 'Fwd Pkts/s':float,
       'Bwd Pkts/s':float , 'Pkt Len Min':int , 'Pkt Len Max':int , 'Pkt Len Mean':float,
       'Pkt Len Std':float , 'Pkt Len Var':float , 'FIN Flag Cnt':int , 'SYN Flag Cnt':int,
       'RST Flag Cnt':int , 'PSH Flag Cnt':int , 'ACK Flag Cnt':int , 'URG Flag Cnt':int,
       'CWE Flag Count':int , 'ECE Flag Cnt':int , 'Down/Up Ratio':float , 'Pkt Size Avg':float,
       'Fwd Seg Size Avg':float , 'Bwd Seg Size Avg':float , 'Fwd Byts/b Avg':float,
       'Fwd Pkts/b Avg':float , 'Fwd Blk Rate Avg':float , 'Bwd Byts/b Avg':float,
       'Bwd Pkts/b Avg':float , 'Bwd Blk Rate Avg':float , 'Subflow Fwd Pkts':float,
       'Subflow Fwd Byts':float , 'Subflow Bwd Pkts':float , 'Subflow Bwd Byts':float,
       'Init Fwd Win Byts':int , 'Init Bwd Win Byts':int , 'Fwd Act Data Pkts':int,
       'Fwd Seg Size Min':int , 'Active Mean':float , 'Active Std':float , 'Active Max':float,
       'Active Min':float , 'Idle Mean':float , 'Idle Std':float , 'Idle Max':float , 'Idle Min':float , 'Label':str}

def get_dtype4normalized():
    return {'Flow ID':str , 'Dst Port':float , 'Protocol':float, 'Timestamp': str,
       'Flow Duration':float , 'Tot Fwd Pkts':float , 'Tot Bwd Pkts': float,
       'TotLen Fwd Pkts':float , 'TotLen Bwd Pkts':float , 'Fwd Pkt Len Max':float,
       'Fwd Pkt Len Min':float , 'Fwd Pkt Len Mean':float , 'Fwd Pkt Len Std':float,
       'Bwd Pkt Len Max':float , 'Bwd Pkt Len Min':float , 'Bwd Pkt Len Mean':float,
       'Bwd Pkt Len Std':float , 'Flow Byts/s':float , 'Flow Pkts/s':float, 'Flow IAT Mean':float,
       'Flow IAT Std':float , 'Flow IAT Max':float , 'Flow IAT Min':float , 'Fwd IAT Tot':float,
       'Fwd IAT Mean':float , 'Fwd IAT Std':float , 'Fwd IAT Max':float , 'Fwd IAT Min':float,
       'Bwd IAT Tot':float , 'Bwd IAT Mean':float , 'Bwd IAT Std':float , 'Bwd IAT Max':float,
       'Bwd IAT Min':float , 'Fwd PSH Flags':float , 'Bwd PSH Flags':float , 'Fwd URG Flags':float,
       'Bwd URG Flags':float , 'Fwd Header Len':float , 'Bwd Header Len':float , 'Fwd Pkts/s':float,
       'Bwd Pkts/s':float , 'Pkt Len Min':float , 'Pkt Len Max':float , 'Pkt Len Mean':float,
       'Pkt Len Std':float , 'Pkt Len Var':float , 'FIN Flag Cnt':float , 'SYN Flag Cnt':float,
       'RST Flag Cnt':float , 'PSH Flag Cnt':float , 'ACK Flag Cnt':float , 'URG Flag Cnt':float,
       'CWE Flag Count':float , 'ECE Flag Cnt':float , 'Down/Up Ratio':float , 'Pkt Size Avg':float,
       'Fwd Seg Size Avg':float , 'Bwd Seg Size Avg':float , 'Fwd Byts/b Avg':float,
       'Fwd Pkts/b Avg':float , 'Fwd Blk Rate Avg':float , 'Bwd Byts/b Avg':float,
       'Bwd Pkts/b Avg':float , 'Bwd Blk Rate Avg':float , 'Subflow Fwd Pkts':float,
       'Subflow Fwd Byts':float , 'Subflow Bwd Pkts':float , 'Subflow Bwd Byts':float,
       'Init Fwd Win Byts':float , 'Init Bwd Win Byts':float , 'Fwd Act Data Pkts':float,
       'Fwd Seg Size Min':float , 'Active Mean':float , 'Active Std':float , 'Active Max':float,
       'Active Min':float , 'Idle Mean':float , 'Idle Std':float , 'Idle Max':float , 'Idle Min':float , 'Label':str, 'Day':str}


def read_data(dataroot, col_names=None, nrows=None):
    
    filenames = [i for i in glob.glob(join(dataroot,'*Meter.csv'))]
    #print("filenames ",filenames)
    if col_names==None:
        col_names = get_cols4ml()
    df_list = []
    for f in filenames:
        print("reading ", ntpath.basename(f))
        if nrows==None:
            df = pd.read_csv(f,usecols=col_names,dtype=get_dtype())
        else:
            df = pd.read_csv(f,usecols=col_names,dtype=get_dtype(),nrows=nrows)
        df_list.append(df)
    combined_csv= pd.concat(df_list,sort=True)
    return combined_csv

def calc_data_statistics(df):
    filename = join(get_ids18_root(),'CSVs','WS_l','data_stats.pickle') 
    columns = np.array(get_cols4norm())
    with open(filename,'wb') as f:
            max_v = df[columns].max()
            min_v = df[columns].min()
            columns = columns[np.where(max_v - min_v > 0)]
            pickle.dump({'columns':columns,'min':df[columns].min(),'max':df[columns].max(),'mean':df[columns].mean()},f)
    

def normalize_df(df):
    filename = join(get_ids18_root(),'CSVs_r_1.0_m_1.0','WS_l','data_stats_allnorm.pickle') 
    #filename = join(get_ids18_root(),'CSVs','WS_l','data_stats.pickle') 
    d = pickle.load(open(filename,'rb'))
    columns = d['columns']
    r = d['max']-d['min']
    df[columns] = (df[columns]-d['mean'])/r
    df[columns] = df[columns].round(4)
    return df


def balance_data(X,y):
    unique,counts = np.unique(y,return_counts=True)
    mean_samples_per_class = int(round(np.mean(counts)))
    new_X = np.empty((0,X.shape[1]))
    new_y = np.empty((0),dtype=int)
    for i,c in tqdm(enumerate(unique)):
        temp_x = X[y==c]
        indices = np.random.choice(temp_x.shape[0],mean_samples_per_class)
        new_X = np.concatenate((new_X,temp_x[indices]),axis=0)
        temp_y = np.ones(mean_samples_per_class,dtype=int)*c
        new_y = np.concatenate((new_y,temp_y),axis=0)

    return (new_X, new_y)


def shuffle(X,y):
    # in order to break class order we need shuffling
    np.random.seed(SEED)
    print('shuffling data with the SEED')    
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    X =  X[indices,:]
    y = y[indices]
    return (X,y)


def extract_sampler_name(d):
    dir_name  = os.path.basename(d)
    sampler_name = dir_name[:dir_name.find('_')]
    if 'SFS' in dir_name:
            color='orange'
            label = 'SFS'
            full_name = 'SketchFlow'
    elif 'SGS' in dir_name:
            color = 'green'
            label = 'SGS'
            full_name = 'Sketch Guided'
    elif 'RPS' in dir_name:
            color = 'blue'
            label = 'RPS'
            full_name = 'Random Packet Sampling'
    elif 'FFS' in dir_name:
            color = 'red'
            label = 'FFS'
            full_name = 'Fast Filtered'
    elif 'SEL' in dir_name:
            color = 'purple'
            label = 'SEL'
            full_name = 'Selective Flow'
    elif 'whole' in dir_name:
            color='black'
            label='W/S'
            full_name = 'Without sampling'
    else:
            print('Investigate plot_comparison',dir_name)
    return label,full_name


def print_processing_time(filename,sampler_dir,N,t):
    if not os.path.isfile(filename):
        with open(filename,'w') as f:
            f.write('{}, {}\n'.format('Sampler','Number of examples (processing time in sec.)' ))

    MCS_IN_SEC = 1000000
    with open(filename,'a+') as f:
        initial,name = extract_sampler_name(sampler_dir)
        f.write("{} ({}), {} ({:.2f})\n".format(name,initial,N,t))

def print_feature_importance(importances,filename):
    col_names = get_cols4ml()
    col_names.remove('Flow ID')
    col_names.remove('Label')

    indices = np.argsort(importances)[::-1]

    print("Printing feature weights")
    with open(filename,'w') as s:
        s.write('Feature, Importance Weight\n')
        for f in range(len(col_names)):
            s.write("{},{:.4f}\n".format(col_names[indices[f]], importances[indices[f]]))


def read_ddos_data(dataroot,columns=None,debug=False):
    # only read common attack in both days
    filenames = ['LDAP.csv','MSSQL.csv','NetBIOS.csv','SYN.csv','UDP.csv','UDP-Lag.csv','records.csv']
    if debug:
        n = 1000
        df_list = [pd.read_csv(join(dataroot,fn),usecols=get_cols4ml(),dtype=get_dtype(), skiprows=lambda x: x%n!=0) for fn in filenames]
    else:
        if columns !=None:
            df_list = [pd.read_csv(join(dataroot,fn),usecols=columns) for fn in filenames]
        
        else:
            df_list = [pd.read_csv(join(dataroot,fn),usecols=get_cols4ml(),dtype=get_dtype()) for fn in filenames]
    if len(df_list)<1:
        print('No file at all, returning')
        return
    combined_csv = pd.concat(df_list,sort=False)
    return combined_csv 

def get_ids18_mappers():
    label_to_id = {'Benign': 0,
        'DDoS-HOIC': 1,
        'DDoS-LOIC-HTTP': 2,
        'DDoS-LOIC-UDP': 3,
        'DoS-GoldenEye': 4,
        'DoS-Hulk': 5,
        'DoS-SlowHTTPTest': 6,
        'DoS-Slowloris': 7,
        'FTP-BruteForce': 8,
        'Infiltration': 9,
        'SSH-BruteForce': 10,
        'Brute Force-Web':11,
        'Brute Force-XSS':12,
        'SQL Injection':13
        }
    id_to_label = {0: 'Benign',
        1: 'DDoS-HOIC',
        2: 'DDoS-LOIC-HTTP',
        3: 'DDoS-LOIC-UDP',
        4: 'DoS-GoldenEye',
        5: 'DoS-Hulk',
        6: 'DoS-SlowHTTPTest',
        7: 'DoS-Slowloris',
        8: 'FTP-BruteForce',
        9: 'Infiltration',
        10: 'SSH-BruteForce',
        11: 'Brute Force-Web',
        12: 'Brute Force-XSS',
        13: 'SQL Injection'
        }
    return label_to_id, id_to_label, list(id_to_label.values())


def get_ddos19_mappers():
    label_to_id = {'Benign': 0, 'LDAP': 1, 'MSSQL': 2, 'NetBIOS': 3, 'SYN': 4, 'UDP': 5, 'UDP-Lag': 6} 
    id_to_label = {0:'Benign',1:'LDAP',2:'MSSQL',3:'NetBIOS',4:'SYN',5:'UDP',6:'UDP-Lag'}
    return label_to_id, id_to_label


def extract_x_y(df):
    y = df['Label'].values
    X_df = df.drop(columns=['Flow ID', 'Label','Timestamp','TimestampMCS','Unnamed: 0','Hash',
    'Src IP', 'Src Port', 'Dst IP','Day'])
    X = X_df.values
    return X,y


def get_ordered_labels():
    return ['Infiltration','SQL Injection', 'Brute Force-XSS', 'Brute Force-Web', 'DDoS-LOIC-UDP', 'DoS-Slowloris', 'DoS-GoldenEye', 'DoS-Hulk', 'DoS-SlowHTTPTest', 'FTP-BruteForce', 'SSH-BruteForce', 'DDoS-LOIC-HTTP', 'DDoS-HOIC']

def get_label_to_csv():
    return {
'Infiltration':['Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv','Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv'],
'SQL Injection':['Friday-23-02-2018_TrafficForML_CICFlowMeter.csv','Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv'],
'Brute Force-XSS':['Friday-23-02-2018_TrafficForML_CICFlowMeter.csv','Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv'],
'Brute Force-Web':['Friday-23-02-2018_TrafficForML_CICFlowMeter.csv','Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv'],
'DDoS-LOIC-UDP':['Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv', 'Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv'],
'DoS-Slowloris':['Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv'],
'DoS-GoldenEye':['Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv'],
'DoS-Hulk':['Friday-16-02-2018_TrafficForML_CICFlowMeter.csv'],
'DoS-SlowHTTPTest':['Friday-16-02-2018_TrafficForML_CICFlowMeter.csv'],
'FTP-BruteForce':['Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv'],
'SSH-BruteForce':['Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv'],
'DDoS-LOIC-HTTP':['Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv'],
'DDoS-HOIC':['Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv']
}


def read_evaldata(fn):
    df = pd.read_csv(fn)

def load_model_n_data(d,label):
    #fn = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/SR_1.0/SFS_SI_95.33_l/10fold_0.csv'
    fn = join(d, '10fold_0.csv')
    df = pd.read_csv(fn, usecols=get_cols4eval())
    
    df = df[df['Label']==label]
    
    df = df.drop(columns=['Flow ID', 'Label'])
    modelfile = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/SI_100/SFS_SI_95.33_l/c_forest_b_explicit_n_100_bootstrap_True_mf_auto_msl_3_ms_0.01_md_25/log/10fold_0.pkl'
    model = pickle.load(open(modelfile,'rb'))
    return [model], [df]

    clf_d = 'c_forest_b_explicit_n_100_bootstrap_True_mf_auto_msl_3_ms_0.01_md_25'
    K = 10
    modelpath_exp = join(d,clf_d,'log','{}fold_{}.pkl'.format(K))
    datapath_exp = join(d,'{}fold_{}.csv')
    models = [pickle.load(open(modelpath_exp.format(i),'rb')) for i in range(K)]
    data = [read_evaldata(datapath_exp.format(K,i)) for i in range(K)]
