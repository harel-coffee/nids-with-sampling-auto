from os.path import join 
import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import pandas as pd
import glob
import time
import ntpath
from tqdm import tqdm
import pickle
from multiprocessing import Pool

from utils import get_ids18_mappers
from utils import get_cols4eval
from utils import get_dtype, get_dtype4normalized
from utils import encode_label
from classifier_settings import get_args, get_classifier
from classifier_settings import ClassifierLoader
from bookkeepers import result_logger_ids18
from classifier_settings import get_classifier_dir, load_classifier


import warnings
warnings.filterwarnings('ignore') 

def group_data(df): # It will prepare the data into classifer usable format
    grouped = df.groupby(['Flow ID','Day','Label'], sort=True) # by default grouped keys are sorted, if df is sorted before grouping, then we can skip ordering part in evaluate per flow
    ID = [ [flowid,day, label]  for (flowid,day, label)  in sorted(grouped.groups.keys())]
    Label = [label for flowid,day,label in ID]
    ID = np.array(ID)
    return ID,Label, grouped


def derive_pred_w_any(num_records_per_flow, y_flow, pred_record):
    flow_level_pred = []
    processed_records = 0
    for fi, num_records_for_the_flow in num_records_per_flow:
        ptr = processed_records
        pred_records_fi = pred[ptr:ptr+num_records_for_the_flow]
        if np.any(y_flow[fi]==pred_flow_records_fi):
            flow_level_pred.append(y_flow[fi])

def evaluate_per_flow_dev(df_grouped_by_fid, y, pred):
    num_record_per_flow = df_grouped_by_fid['Flow ID'].agg('count')
    any_pred = derive_pred_w_any()



def evaluate_per_flow(grouped, y, pred): #please read the paper for the difference between flow and flow record
    record_count_ls = grouped['Label'].agg('count')

    flow_pred_any = []
    flow_pred_majority = []
    flow_pred_all = []


    p_records = 0 # pointer for processed_records
    tick = time.time()
    for i,n_records in (enumerate(record_count_ls)):# number of records for each flow
        pred_i = pred[p_records:p_records+n_records]
        counts = np.bincount(pred_i)
        most_common_pred = np.argmax(counts)
        if 'any'=='any':
            if (pred_i==y[i]).sum()>0:
                flow_pred_any.append(y[i]) # if any of the records are detected, we consider flow is detected
            else:
                flow_pred_any.append(most_common_pred) # else, most common misprediction label is given to flow
        if 'majority'=='majority':
            flow_pred_majority.append(most_common_pred) # most common prediction label is given to flow
        if 'all'=='all':
            if np.all(y[i]==pred_i):
                flow_pred_all.append(y[i])
            else:
                error_counts = np.bincount(pred_i[np.where(pred_i!=y[i])])
                most_common_error = np.argmax(error_counts)
                flow_pred_all.append(most_common_error)
        p_records+=n_records
    tock = time.time()
    eval_duration = tock-tick
    #print("Time for Evaluation: {:.0f} min {} sec".format((eval_duration)//60,(eval_duration)%60)) # 80min WS 

    return np.array(flow_pred_any),np.array(flow_pred_majority), np.array(flow_pred_all)


BENIGN_CLASS_ID = 0
def predict_per_record(df, clf):
        X = df.drop(columns=['Flow ID','Day','Label','Timestamp']).values
        # overcome memory limitation by predicting chunk-by-chunk
        chnksz = X.shape[0]//5
        pred = np.concatenate([clf.predict(X[i:i+chnksz]) for i in range(0,X.shape[0],chnksz)],axis=0)
        return pred


def predict_proba_per_record(df,clf, benign_threshold=None):
        X = df.drop(columns=['Flow ID','Day','Label','Timestamp']).values
        # overcome memory limitation by predicting chunk-by-chunk
        chnksz = X.shape[0]//5 
        prob = np.concatenate([clf.predict_proba(X[i:i+chnksz]) for i in range(0,X.shape[0],chnksz)],axis=0)
        prob[prob[:,BENIGN_CLASS_ID]>=benign_threshold,BENIGN_CLASS_ID]=1.1 # set benign proba to zero if it is less than threshold
        pred = np.argmax(prob, axis=1)
        pred = np.fromiter(map(lambda x: clf.classes_[x],pred), dtype=np.int) # revert to clf labels 
        return pred


def per_record_evaluation(df,pred):
        y_per_record = np.array(encode_label(df.Label.values))
        acc_per_record = metrics.balanced_accuracy_score(y_per_record,pred)
        print("----------per record acc: {:.2f}-----------".format(acc_per_record))


def replace_w_unlimited_FC(runs_dir):
            start = runs_dir.find('CSVs_r')
            end = runs_dir.find('_m_')
            CSV_dirname = runs_dir[start:end]
            return runs_dir.replace(CSV_dirname, 'CSVs_r_1.0')


def evaluator(args):
    is_flow_cache_experiment = True
    K = 10 
    samplerdir,classifier_name, benign_threshold = args
    print('treshold at ', benign_threshold)
  
    clf_dir = get_classifier_dir(samplerdir, classifier_name, class_weight=None)
    gt_classes_str = pd.read_csv(join(samplerdir,'{}fold_0.csv'.format(K)),usecols=['Label'])['Label'].unique()
    gt_classes = sorted(encode_label(gt_classes_str))

    
    C = len(gt_classes)
    cm_any_sum = np.zeros((C,C),dtype=float)
    cm_majority_sum = np.zeros((C,C),dtype=float)
    cm_all_sum = np.zeros((C,C),dtype=float)
    
    col_names = ['Timestamp']+get_cols4eval()
    for test_index in range(K):
        runs_dir = join(clf_dir,'K_{}/log/{}'.format(K, test_index)) 
        if is_flow_cache_experiment:
            runs_dir = replace_w_unlimited_FC(runs_dir)
        clf =  load_classifier(classifier_name, runs_dir)       
        
        test_csv_file = join(samplerdir, '{}fold_{}.csv'.format(K,test_index))
        df = pd.read_csv(test_csv_file,usecols=col_names,dtype=get_dtype4normalized())#,skiprows=skip_idx)
        df['Day'] = df['Timestamp'].map(lambda x: x[:2]).astype(str) # type string 
        df = df.sort_values(by=['Flow ID','Day','Label']) #used when deriving flow level metric

        pred_per_record = predict_proba_per_record(df, clf, benign_threshold )
        flowids, flowlabels_str, grouped = group_data(df)
  
        y = encode_label(flowlabels_str)
        pred_any, pred_maj, pred_all = evaluate_per_flow(grouped, y, pred_per_record)

        any_cm = confusion_matrix(y, pred_any)
        majority_cm = confusion_matrix(y,pred_maj)
        all_cm = confusion_matrix(y,pred_all)
        
        cm_any_sum+=any_cm
        cm_majority_sum += majority_cm
        cm_all_sum += all_cm
        #gt_classes = np.unique(y)
        result_logger_ids18(join(clf_dir,'K_{}_benign_threshold_{}'.format(K,benign_threshold)), gt_classes,(any_cm, majority_cm, all_cm), 'fold_{}_'.format(test_index))
    result_logger_ids18(join(clf_dir,'K_{}_benign_threshold_{}'.format(K,benign_threshold)), gt_classes,(cm_any_sum, cm_majority_sum, cm_all_sum), 'fold_avg_'.format(K))

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
        samp_inter = 'SI_10'
        classifiers =  ['cnn','tree', 'forest']
        ben_thresholds = np.geomspace(0.01,1,10).round(3)
        for csv_dir in csv_dirs[:1]:
            for d in sampler_dirs[samp_inter]:
                datadir = '/data/juma/data/net_intrusion/ids18/{}/{}/{}'.format(csv_dir,samp_inter,d)
                for clf in classifiers[1:]:
                    list_of_args = [[datadir, clf, t] for t in ben_thresholds]
                    chnksz =3 
                    for i in range(0,len(list_of_args),chnksz):
                        list_of_chnk = list_of_args[i:i+chnksz] 
                        with Pool(chnksz) as p:
                            p.map(evaluator,list_of_chnk)
                    
