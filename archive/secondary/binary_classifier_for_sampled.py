import os
from os.path import join as os_join
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from utils import make_value2index
from CICIDS2017 import read_data
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
import operator
import argparse
from utils import get_complete_labels_d, get_complete_label_names, get_bad_attacks, read_csv
import csv
from sklearn.utils import resample

#dataroot = '/home/jumabek/IntrusionDetectionSampling/data/GeneratedLabelledFlows/sampled/TrafficLabelling_RS_SR10'
#dataroot = '/home/isrl/data/CIC-IDS-2017/sampled_data/sFlowSR10'
SEED = 234
parser = argparse.ArgumentParser()
parser.add_argument('-dataroot', default = '/home/isrl/data/CIC-IDS-2018/sampled_data/sk_sr10_l', 
                        help='path to dataroot folder\n' )
parser.add_argument('-classifier', default='forest', help = 'which classifier to use')

args = parser.parse_args()
dataroot = args.dataroot
log_file = os_join(dataroot,  args.classifier + "_filtered_results.txt")
K=5


def get_binary_labels():
    return {"BENIGN":0,"Malicious":1}


def get_finegrained_labels(arr):
    Y_str = arr[:,-1]
    unique_y_str = np.unique(Y_str)
    fine_grained_labels = make_value2index(unique_y_str)
    return fine_grained_labels


def print_labels():
    print('sampled train set labels are:')
    for key in sorted(labels.keys()):
        print ("%s: %s" % (key, labels[key]))


def print_label_statistics(data,split=None,stream=None):
    global args
    dataroot = args.dataroot
    labels = get_complete_labels_d()
    #labels = sorted(labels.items(), key=operator.itemgetter(1))
    print(labels)
    print('{} labels '.format(split), labels,file=stream)
    y = data[:,-1]   
    index_row = []
    label_row = []
    count_row = []
       
    for label,index in sorted(labels.items(), key=lambda kv: kv[1]):
        c = np.count_nonzero(y==label)
        index_row.append(index)
        label_row.append(label)
        count_row.append(c)
        print("{:2}, {:30}, {:10}".format(index,label,c),file=stream)
    
    with open(os_join(dataroot,"distribution.csv"),"w") as f:    
        w = csv.writer(f,delimiter=',')
        w.writerow(label_row)
        w.writerow(index_row)
        w.writerow(count_row)


def correct_data(arr, bad_labels,FINEGRAINED=False): # It will prepare the data into classifer usable format
    # please look at the code
    labels = get_complete_labels_d()
    num_cols = 32
    if arr.shape[1]==num_cols:
        filtered_indices = [i for i in np.arange(num_cols) if i not in [0,1,2,3,5,6]]
        arr = arr[:,filtered_indices]

    X = arr[:,:-1].astype(np.float32)
    Y_str = arr[:,-1]
    rows_without_nan = ~np.isnan(X).any(axis=1)
    X = X[rows_without_nan]
    Y_str = Y_str[rows_without_nan]

    rows_with_finite = np.isfinite(X).all(axis=1)
    X = X[rows_with_finite]
    Y_str = Y_str[rows_with_finite]

    # thanks to the sampling, it is possible that test/val set might have a label that train set do not have.
    # here we mitigate this problem

    bad_flow_indices = []
    for index,y_str in enumerate(Y_str):
        if y_str in bad_labels or y_str not in labels:
            bad_flow_indices.append(index)
    Y_str = np.delete(Y_str,bad_flow_indices,axis=0)
    X = np.delete(X,bad_flow_indices,axis=0)

    if not FINEGRAINED:
        return X,Y_str
    #remove classes which has less than K samples, to make sure stratified K-fold works
    unique,counts = np.unique(Y_str,return_counts=True)
    for i,count in enumerate(counts):
        if count<K:
            indices = Y_str!=unique[i]
            Y_str = Y_str[indices]
            X = X[indices]

    return X,Y_str


def encode_label(Y_str, FINEGRAINED=False): 
    labels = get_complete_labels_d()
    if FINEGRAINED: # fine grained labels
        Y = [labels[y_str] for y_str  in Y_str]
        Y = np.array(Y)
    else:
        Y = np.ones(Y_str.shape)
        Y[Y_str=='BENIGN'] = 0
    return np.array(Y)


def get_classifier():
    global args
    classifier = args.classifier
    if classifier=='tree':
        return tree.DecisionTreeClassifier()
    elif classifier=='forest':
        return RandomForestClassifier()
    elif classifier=='linear_svm':
        return svm.SVC(gamma='scale')
 

def print_cm(cm,label_names,stream):
    # label1 00,01,02,..
    # label2 10,11,12,..
    # ...
    for i,label in enumerate(label_names):
        l = cm[i,:].astype(np.str_)
        s = ""
        for item in l:
            s+="{:8} ".format(item)
        print('{:30} {}'.format(label,s),file=stream)
    return  


def print_evaluation(cm,label_names,stream):
    print("\n\nEvaluation:",file=stream)
    for i,label in enumerate(label_names):
        tp = cm[i,i]
        fp = np.sum(cm[:,i]) - cm[i,i]
        fn = np.sum(cm[i,:]) - cm[i,i]  
        pr = tp/(fp+tp)
        rc = tp/(fn+tp)
        f1 = 2*pr*rc/(pr+rc)
        print("{:30}  {:8.3f}{:8.3f}{:8.3f}".format(label,pr,rc,f1),file=stream)  


def read_data(dataroot):
    filenames = get_filenames(dataroot)
    data = read_csv(os_join(dataroot,filenames[0]))
    for filename in filenames[1:]:
       data_i = read_csv(os_join(dataroot,filename))
       data = np.concatenate((data,data_i),axis=0)
    return data


def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir,name)) and not name.startswith(".~lock.") and (name.endswith(".pcap_ISCX.csv") or name.endswith(".pcap_Flow.csv"))]


data = read_data(dataroot)
with open(log_file,"w") as f:
    print_label_statistics(data,'all',stream=f)

X,Y = correct_data(data,get_bad_attacks())
Y = encode_label(Y,FINEGRAINED=True)
print("data shape",X.shape, Y.shape)
print("Unique labels after data correction ",np.unique(Y))

clf = get_classifier() 
skf = StratifiedKFold(n_splits=K,random_state=SEED)

unique,counts = np.unique(Y,return_counts=True)
print(np.asarray((unique,counts)).T)
num_class = len(np.unique(Y))
confusion_matrix_sum = np.zeros((num_class, num_class),dtype=int)
for train_index,test_index in skf.split(X,Y):
    clf.fit(X[train_index],Y[train_index])
    pred = clf.predict(X[test_index])
    print(np.unique(Y[test_index]))
    print(confusion_matrix_sum.shape,confusion_matrix(Y[test_index],pred).shape)
    confusion_matrix_sum += confusion_matrix(Y[test_index],pred )

with open(log_file,"a") as f:
    print("\nCONFUSION MATRIX: \n",file=f)
    cm = np.array(confusion_matrix_sum/K).astype(np.int)
    unique_label_ids = np.unique(Y)
    label_names = get_complete_label_names()
    unique_label_names = [label_names[id] for id in unique_label_ids]
    print_cm(cm,unique_label_names,stream=f)
    print_evaluation(cm, unique_label_names,stream=f)

# Now for the binary part
X,Y = correct_data(data,get_bad_attacks())
Y = encode_label(Y)
print("data shape",X.shape, Y.shape)
clf = get_classifier()
skf = StratifiedKFold(n_splits=K,random_state=SEED)

num_class = len(np.unique(Y))
confusion_matrix_sum = np.empty((num_class,num_class),dtype=np.float64)
for train_index,test_index in skf.split(X,Y):
    x = X[train_index]
    y = Y[train_index]
    clf.fit(x,y)
    pred = clf.predict(X[test_index])
    confusion_matrix_sum += confusion_matrix(Y[test_index],pred) 

with open(log_file,"a") as f:
    print("\n\nCONFUSION MATRIX:\n", file=f)
    cm = np.array(confusion_matrix_sum/K).astype(np.int)

    unique_label_names = ['Benign','Attack']
    print_cm(cm,unique_label_names,stream=f)

    TP = cm[1,1]
    FN = cm[1,0]
    FP = cm[0,1]
    pr = TP/(TP+FP)
    rc = TP/(TP+FN)
    F1 = 2*pr*rc/(pr+rc)
    print('Precision: {:.3} | Recall: {:,.3} | F1:{:.3}'.format(pr,rc,F1),file=f)



