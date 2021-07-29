from os.path import join
import numpy as np
from sklearn import metrics
import pandas as pd
import time
from tqdm import tqdm

from utils import ensure_dir,ensure_dirs
from utils import normalize_df, shuffle
from utils import print_processing_time, print_feature_importance
from utils import get_dtype, get_cols4ml, read_ddos_data
from utils import get_ddos19_mappers
from classifier_settings import get_args, get_balancing_technique, ClassifierLoader
from bookkeepers import result_logger_ddos19
SEED = 234

def df_to_array(df):
    y = df['Label'].values
    X_df = df.drop(columns=['Flow ID', 'Label'])
    X = X_df.values
    return X,y


def encode_label(Y_str,labels_d):
    return [labels_d[y_str] for y_str  in Y_str]


def group_data(df): # It will prepare the data into classifer usable format
    grouped = df.groupby(['Flow ID','Label'],sort=True)
    n_counts = grouped.size()
    
    ID = [ [flowid,label]  for (flowid,label)  in grouped.groups.keys()]
    Label = [label for flowid,label in ID]
    ID = np.array(ID)
    return ID,Label,grouped,n_counts


def predict_per_flow(classifier_name,clf,grouped,ordered_df,y_true,group_sizes):
    flow_pred_any = []
    flow_pred_majority = []
    flow_pred_all = []
    pred_i = None

    X = ordered_df.values
    print('Extracted X with shape ',X.shape)

    tick = time.time()
    if classifier_name in ['cnn','softmax']:
        pred = clf.predict(X,eval_mode=True)
    else:
        pred = clf.predict(X)
    del ordered_df
    tock = time.time()
    prediction_time = tock - tick

    p_records = 0 # processed_records

    for i,n_records in tqdm(enumerate(group_sizes)):
        y = y_true[i]
        #print(i,n_records):
        pred_i = pred[p_records:p_records+n_records]
        counts = np.bincount(pred_i)
        most_common_pred = np.argmax(counts)
        if 'any'=='any':
            if (pred_i==y).sum()>0:
                flow_pred_any.append(y) # if any of the records are detected, we consider flow is detected
            else:
                flow_pred_any.append(most_common_pred) # else, most common misprediction label is given to flow
        if 'majority'=='majority':
            flow_pred_majority.append(most_common_pred) # most common prediction label is given to flow
        if 'all'=='all':
            if np.all(y==pred_i):
                flow_pred_all.append(y)
            else:
                error_counts = np.bincount(pred_i[np.where(pred_i!=y)])
                most_common_error = np.argmax(error_counts)
                flow_pred_all.append(most_common_error)
        p_records+=n_records
    tock = time.time()
    print("Evaluation time: {:2.0f}:{:2.0f} sec".format((tock-tick)//60,(tock-tick)%60))
    return flow_pred_any, flow_pred_majority, flow_pred_all, prediction_time


def evaluate(traindir,testdir,classifier_name):
    pred_any_list = []
    pred_majority_list = []
    pred_all_list = []
    y_test_perflowid_list = []

    pre_fingerprint = join(traindir, 'c_{}'.format(classifier_name))
    balancing_technique = get_balancing_technique() 
    label_to_id, id_to_label = get_ddos19_mappers()

    filenames = ['LDAP.csv','MSSQL.csv','NetBIOS.csv','SYN.csv','UDP.csv','UDP-Lag.csv','records.csv']

    total_prediction_time = 0
    total_records         = 0

    for fn in filenames:
        print("---------------------------")
        print("Reading {}".format(fn))
        tick = time.time()
        test_df = pd.read_csv(join(testdir,fn),usecols=get_cols4ml()) #read in 2min, requires 14GB memory
        tock = time.time()
        input_dim = test_df.shape[1]-2 # flow id and Label is dropped
        num_class = len(label_to_id.keys())
        print("Read {} records in {:.2f} min".format(test_df.shape[0],(tock-tick)/60.))
        if test_df.shape[0]<1:
            continue 
        test_df = test_df.sort_values(by=['Flow ID','Label'])# makes grouping,faster. Allows predict per flowid  
        dummy_num_records = test_df.shape[0]
        class_weight = None
        classifier_args,config = get_args(classifier_name,dummy_num_records,num_class,input_dim, class_weight,balancing_technique)
        # directories for results
        train_fingerprint = join(traindir,'c_{}'.format(classifier_name + config)) # fingerprint already there
        logdir = join(train_fingerprint,'log') #already there
        runs_dir = join(logdir,'runs')
        test_df = normalize_df(test_df,join(runs_dir,'data_stats.pickle'))
        
        fingerprint = join(testdir,'c_{}'.format(classifier_name + config)) # fingerprint already there
        #create classifier
        loader = ClassifierLoader()
        classifier_args['runs_dir'] = runs_dir
        clf = loader.load(classifier_args)

        # predict part
        print("Grouping data \r")
        tick = time.time()
        test_flowids,y_test_perflowid_str, grouped, group_sizes = group_data(test_df)
        test_df = test_df.drop(columns=['Flow ID','Label'])
        tock = time.time()
        print("Done. In {:.0f}min".format((tock-tick)/60.))
        
        y_test_perflowid = encode_label(y_test_perflowid_str,label_to_id)
        
        pred_any, pred_majority, pred_all, prediction_time = predict_per_flow(classifier_name,clf,grouped,test_df,y_test_perflowid, group_sizes) # takes 2-3 min
        
        total_prediction_time += prediction_time
        total_records         += test_df.shape[0]

        pred_any_list +=pred_any
        pred_majority_list +=pred_majority
        pred_all_list += pred_all

        y_test_perflowid_list += y_test_perflowid 
    
    
    pd.DataFrame({'Records':[total_records],'Time':[total_prediction_time]}).to_csv(join(testdir,'timing.csv'),index=False)
    pred_list_tuples = (pred_any_list, pred_majority_list, pred_all_list)
    result_logger_ddos19(fingerprint,y_test_perflowid_list,pred_list_tuples,id_to_label) 


if __name__ == '__main__':
    traindir = '/data/juma/data/ddos/CSVs_r_1.0/SR_10/SGS_e_0.0017/PCAP-01-12_l'
    #testdir = traindir
    testdir = '/data/juma/data/ddos/CSVs_r_1.0/SR_10/SGS_e_0.0028/PCAP-03-11_l'
    tick = time.time()
    predict(traindir, testdir,'cnn')# took 26min for WS
    tock = time.time()
    print("Total time it took for module is {:.0f}".format(tock-tick))
