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
from utils import get_cols4eval, get_dtype, get_dtype4normalized
from utils import encode_label
from classifier_settings import get_args, get_classifier
from classifier_settings import ClassifierLoader
from bookkeepers import result_logger_ids18


def group_data(df): # It will prepare the data into classifer usable format
    print("Grouping")
    grouped = df.groupby(['Flow ID','Day','Label'], sort=True) # by default grouped keys are sorted, if df is sorted before grouping, then we can skip ordering part in evaluate per flow
    ID = [ [flowid,day, label]  for (flowid,day, label)  in tqdm(sorted(grouped.groups.keys()))]
    Label = [label for flowid,day,label in ID]
    ID = np.array(ID)
    return ID,Label, grouped


def evaluate_per_flow(clf,y, grouped, ordered_df, pred):
    print("Getting counts per group", end='\r')
    tick = time.time()

    record_count_ls = grouped['Label'].agg('count')
    #X = ordered_df.drop(columns=['Flow ID','Label','Day','Timestamp']).values
    print('Prepared Input in : {:.2f} '.format(time.time()-tick))# 1M => 35 min, 30M=>1000min= 17 hours

    # prediction
    print("Start prediction")
    tick = time.time()
    #pred = clf.predict(X)
    tock = time.time()
    prediction_time = tock - tick
    print("Predicted in {:.2f}".format(prediction_time))

    # evaluate prediction per-flow (5 tuple)
    flow_pred_any = []
    flow_pred_majority = []
    flow_pred_all = []
    p_records = 0 # processed_records
    tick = time.time()
    for i,n_records in tqdm(enumerate(record_count_ls)):
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
    print("Time for Evaluation: {:.0f} min {} sec".format((eval_duration)//60,(eval_duration)%60)) # 80min WS 

    return np.array(flow_pred_any),np.array(flow_pred_majority), np.array(flow_pred_all), y


def predict_per_record(df,clf):
        x_per_record = df.drop(columns=['Flow ID','Day','Label','Timestamp']).values
        N=x_per_record.shape[0]
        pivot = N//3
        pred1 = clf.predict(x_per_record[:pivot])
        pred2 = clf.predict(x_per_record[pivot:2*pivot])
        pred3 = clf.predict(x_per_record[2*pivot:])

        pred =  np.concatenate((pred1,pred2,pred3),axis=0)
        return pred


def per_record_evaluation(df,pred_per_record):
        print("----------per record analyusis-----------")
        y_per_record = encode_label(df.Label.values) 
        
        acc_per_record = metrics.balanced_accuracy_score(y_per_record,pred_per_record)
        print(acc_per_record)
        print("end of per_record analysis")


def get_num_test_records(csv_file):
        result_test = subprocess.run(['wc','-l',test_csv_file], stdout=subprocess.PIPE)
        test_records = int(result_test.stdout.split()[0])
        print("# test records: ",test_records)
        return test_records 


def evaluator(dataroot,classifier_name):
        print(ntpath.basename(dataroot))
        test_csv_file = join(dataroot,'fold_0.csv')

        # load Classifier
        classifier_args, config = get_args(classifier_name, num_class='dummy', class_weight=None )
        print("Balancing technique: ", classifier_args['balance'])
        pre_fingerprint = join(dataroot, 'c_{}'.format(classifier_name))
        fingerprint = pre_fingerprint + config
        logdir = join(fingerprint,'log')

        if 'mem_const_exp' == 'mem_const_exp':
            # for mem constraint exp
            start = logdir.find('CSVs_r')
            end = logdir.find('_m_')
            CSV_dirname = logdir[start:end]
            logdir = logdir.replace(CSV_dirname, 'CSVs_r_1.0')
            # end

        print(logdir)
        classifier_args['runs_dir'] = logdir

        loader = ClassifierLoader()
        clf = loader.load(classifier_args)
        
        if 'noprint_clf_attr'=='print_clf_attr' and 'tree' in classifier_name:
            print("maximum depth of the tree ", clf.tree_.max_depth)
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree
            plt.figure()
            plot_tree(clf,filled=True)
            plt.savefig(join(logdir,'tree_plot.png'),dpi=1000)        
            return
        if 'norf_attr'=='rf_attr' and 'forest' in classifier_name:
            depth = [est.tree_.max_depth for est in clf.estimators_ ] 
            print(depth)
            depth = np.array(depth)
            print("forest depth",depth.mean(),depth.max(), depth.min())
            print("maximum depth of the tree ", clf.base_estimator_.max_depth)
            return
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree
            plt.figure()
            plot_tree(clf,filled=True)
            plt.savefig(join(logdir,'tree_plot.png'),dpi=1000)        
            return


        print("Classifier Loaded!")
        # classifier loaded

        # load data
        col_names = get_cols4eval()
        col_names.append('Timestamp')
        df = pd.read_csv(test_csv_file,usecols=col_names,dtype=get_dtype4normalized())#,skiprows=skip_idx)
        df['Day'] = df['Timestamp'].map(lambda x: x[:2]).astype(str) # type string 
        df = df.sort_values(by=['Flow ID','Day','Label'])
        print(df.Label.value_counts())
  
        # Done    
        pred_per_record = predict_per_record(df, clf)
        per_record_evaluation(df,pred_per_record)
        
        tick = time.time()
        flowids, flowlabels_str, grouped = group_data(df)
        print("Grouped in {:.2f} min".format((time.time()-tick)/60))
        y = encode_label(flowlabels_str)
        print("data is grouped and labels are encoded")

        pred_any, pred_maj, pred_all, y = evaluate_per_flow(clf,y,grouped,df,pred_per_record)
        
        gt_classes = np.unique(y)
        pred_classes = np.unique(pred_any)
        nunique_gt = len(gt_classes)
        nunique_pred = len(pred_classes)

        assert nunique_gt >= nunique_pred,"should not predict non existing class(es), but \n{} < \n{}".format(gt_classes, pred_classes)
        any_cm = confusion_matrix(y, pred_any)
        majority_cm = confusion_matrix(y,pred_maj)
        all_cm = confusion_matrix(y,pred_all)
        
        any_acc = metrics.balanced_accuracy_score(y,pred_any)
        maj_acc = metrics.balanced_accuracy_score(y,pred_maj)
        all_acc = metrics.balanced_accuracy_score(y,pred_all)
        print(any_acc, maj_acc, all_acc)
        result_logger_ids18(fingerprint, gt_classes,(any_cm, majority_cm, all_cm), 'test')


if __name__=='__main__':
    if 'WS'=='WS':
        #datadir = '/data/juma/data/ids18/CSVs/WS_l'
        datadir = '/hdd/juma/data/net_intrusion/ids18/CSVs_r_1.0_m_1.0/WS_l'
        evaluator(datadir, 'tree')
    else:
        csv_dir = 'CSVs_r_0.01_m_1.0'
        #csv_dir = 'CSVs_r_1.0_m_1.0'
        #sampler_dirs = ['SFS_SI_9.77_l', 'SGS_e_0.05_l','SRS_SI_10_l','FFS_(8,16,4)_l']
        sampler_dirs = ['SFS_SI_95.33_l', 'SGS_e_1_l','SRS_SI_100_l','FFS_(8,16,40)_l']
        #sampler_dirs = ['SFS_SI_685.08_l','SRS_SI_1000_l','SGS_e_11.5_l','FFS_(8,16,400)_l']
        sdir = 'SR_1.0'
        for d in sampler_dirs:
            datadir = '/data/juma/data/ids18/{}/{}/{}'.format(csv_dir,sdir,d)
            evaluator(datadir,'forest')
