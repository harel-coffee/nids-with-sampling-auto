import os
from os.path import join 
import numpy as np
import pandas as pd
import glob
import time
import ntpath
from tqdm import tqdm

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from bookkeepers import result_logger_ids18 
from utils import print_feature_importance
from utils import get_ids18_mappers
from utils import extract_x_y
from classifier_settings import get_args, ClassifierLoader, get_balancing_technique
from make_fold_by_chunk import get_flowids_and_labels


SEED = 234
def get_class_weights(y):
    unique, counts = np.unique(y,return_counts=True)
    weight = 1./counts # make it probabilyt where frequent class has smaller weight
    s = sum(weight)
    norm_weight = weight/s  
    return norm_weight


def group_data(df): # It will prepare the data into classifer usable format
    grouped = df.groupby(['Flow ID','Day','Label'])
    ID = [ [flowid,day, label]  for (flowid,day, label)  in grouped.groups.keys()]
    Label = [label for flowid,day,label in ID]
    ID = np.array(ID)
    return ID,Label, grouped


def encode_label(Y_str,labels_d): 
    Y = [labels_d[y_str] for y_str  in Y_str]
    Y = np.array(Y)
    return np.array(Y)


def predict_fold(classifier_name,clf,df,y_true, grouped, dataroot):
    flow_pred_any = []
    flow_pred_majority = []
    flow_pred_all = []

    tick = time.time()
    ordered_df = pd.concat([frames for ((flowid,day,label),frames) in tqdm(grouped)])
    ordered_df = ordered_df.drop(columns=['Flow ID','Label','Timestamp','TimestampMCS','Unnamed: 0','Hash','Src IP', 'Src Port','Dst IP','Day'])
    X = ordered_df.values
    print('Preparing Input in ordered fashion: {:.2f} '.format(time.time()-tick))

    # prediction
    print("Start prediction")
    tick = time.time()
    if classifier_name in ['cnn','softmax']:
        pred = clf.predict(X,eval_mode=True)
    else:
        chunk_size = 10**4
        n_chunks = X.shape[0]//chunk_size+1*(X.shape[0]%chunk_size)
        for ch in tqdm(range(n_chunks)):
            pred_chunk = clf.predict(X[ch*chunk_size:(ch+1)*chunk_size])
            if ch==0:
                pred = pred_chunk
            else:
                pred = np.concatenate((pred,pred_chunk),axis=0)
            
    tock = time.time()
    prediction_time = tock - tick
    #print("Predicted data of size {}(samples/second) in {:.2f} sec".format(X.shape[0], prediction_time))
    
    # evaluate prediction per-flow (5 tuple)
    p_records = 0 # processed_records
    tick = time.time()
    for i,((flowid,day,label),frames) in tqdm(enumerate(grouped)):
        #print(frames.columns)
        #X_i = frames.drop(columns=['Flow ID','Label','Timestamp','TimestampMCS','Unnamed: 0','Hash','Src IP', 'Src Port','Dst IP','Day']).values
        #pred_i = clf.predict(X_i)

        y = y_true[i]
        n_records = frames.shape[0] # number of records in a flow
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
    eval_duration = tock-tick
    print("Time for Evaluation: {:.0f} min {} sec".format((eval_duration)//60,(eval_duration)%60))

    return np.array(flow_pred_any),np.array(flow_pred_majority), np.array(flow_pred_all), eval_duration


def classify(dataroot,classifier_name):
        K=5
        fraction = 1 
        label_to_id, id_to_label, _  = get_ids18_mappers()
        #class_weight = get_class_weights(encode_label(total_label_df.values,label_to_id))
        class_weight = None

        balance = get_balancing_technique()
        input_dim = 78 # because we remove Label and FlowID columns from X
        gt_num_class = len(label_to_id)

        classifier_args,config = get_args(classifier_name, gt_num_class, input_dim,class_weight, balance)
        pre_fingerprint = join(dataroot, 'r_{}_c_{}_k_{}'.format(fraction,classifier_name,str(K)))
            
        fingerprint = pre_fingerprint + '_mem_constrained' + config
        logdir = join(pre_fingerprint+config,'log')

        cm_any = np.zeros((gt_num_class, gt_num_class),dtype=float)
        cm_majority = np.zeros((gt_num_class, gt_num_class),dtype=float)
        cm_all = np.zeros((gt_num_class, gt_num_class),dtype=float)
        
        kfold_feature_importance = np.zeros(input_dim,dtype=np.float)
        for fold_index in range(K):
            print('###################################')
            print("Fold ",fold_index)
            test_df = pd.read_csv(join(dataroot,'fold_{}.csv'.format(fold_index))) 
            runs_dir=join(logdir,'fold_{}'.format(fold_index))
            # for mem constrained experiemnt II, we need same classifier CSVs_r_1 for all memories
            start = runs_dir.find('CSVs_r_')
            end = runs_dir.find('SR_10')
            CSV_dirname = runs_dir[start:end-1]
            #runs_dir = runs_dir.replace(CSV_dirname,'CSVs_r_1.0')
            classifier_args['runs_dir'] = runs_dir
            #----------------
            loader = ClassifierLoader()
            clf = loader.load(classifier_args)
            print("Loaded Classifier!")
            if classifier_name=='forest':
                kfold_feature_importance+=clf.feature_importances_
            
            flowids_test,y_flowid_test, grouped = group_data(test_df)
            y_flowid_test = encode_label(y_flowid_test,label_to_id)
            pred_any, pred_majority, pred_all, duration = predict_fold(classifier_name,clf,test_df,y_flowid_test,grouped,dataroot)
            assert pred_any.shape==pred_majority.shape,"any and majority shapes should be same {},{}".format(pred_any.shape,pred_majority.shape)
            
            acc_pred_any = 100*metrics.balanced_accuracy_score(y_flowid_test,pred_any)
            acc_pred_majority = 100*metrics.balanced_accuracy_score(y_flowid_test,pred_majority)
            acc_pred_all = 100*metrics.balanced_accuracy_score(y_flowid_test,pred_all)
            print("Fold Local Balanced accuracy(any,majority,all): ({:.2f},{:.2f},{:.2f})".format(acc_pred_any,acc_pred_majority, acc_pred_all))

            any_cm_i = confusion_matrix(y_flowid_test,pred_any)
            majority_cm_i = confusion_matrix(y_flowid_test,pred_majority)
            all_cm_i = confusion_matrix(y_flowid_test,pred_all)
            
            result_logger_ids18(fingerprint, y_flowid_test,(any_cm_i, majority_cm_i, all_cm_i), id_to_label, str(fold_index)+'_') 
            
            cm_any+=any_cm_i
            cm_majority+=majority_cm_i
            cm_all+=all_cm_i
        if classifier_name=='forest':
            print_feature_importance(kfold_feature_importance,join(dataroot,'folds_fraction_{}'.format(fraction),'feature_selection.csv'))

        print(dataroot,classifier_name)
        result_logger_ids18(fingerprint, y_flowid_test,(cm_any, cm_majority, cm_all), id_to_label, 'avg_') 


if __name__=='__main__':
    #datadir = '/data/juma/data/ids18/CSVs_r_1.0/SR_10/RPS_SI_10_l'
    datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs/WS_l' 
    classify(datadir,'forest')
