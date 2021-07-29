from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import pandas as pd

from utils import ensure_dirs, ensure_dir
from utils import get_ddos19_root, get_ids18_root
from utils import get_ids18_mappers


def plot_confusion_matrix(filename,y_true, y_pred, classes,id_to_label,cm=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    np.set_printoptions(precision=2)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    if cm is None:
        cm = confusion_matrix(y_true, y_pred)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots(figsize=(20,16))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if normalize:
        fmt = '.2f'
        thresh = (cm_norm.max()+cm_norm.min()) / 2.
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, format(cm_norm[i, j], fmt), horizontalalignment='center',
                        verticalalignment='center',
                        color="white" if cm_norm[i, j] > thresh else "black")
    else:
        fmt = '.0f'
        thresh = (cm_norm.max()+cm_norm.min()) / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                        verticalalignment='center',
                        color="white" if cm_norm[i, j] > thresh else "black")


    fig.tight_layout()
    plt.savefig(join(filename),dpi=200)
    plt.close(fig)


    txt_filename = filename.replace('.jpg','.csv')
    f = open(txt_filename,'w')
    f.write('{}'.format(''))
    for cname in classes:
        f.write(',{}'.format(cname))
    f.write('\n')

    # Loop over data dimensions and create text annotations.
    if normalize:
        cm = cm_norm
    for i in range(cm.shape[0]):
        f.write('{}'.format(id_to_label[i]))
        for j in range(cm.shape[1]):
            f.write(',{0:.4f}'.format(cm[i,j]))
        f.write('\n')
    f.close()

    return ax


def print_evaluation(cm,cm_labels,output_dir,fold_index, det_policy):
    cm_labels = np.array(cm_labels)
    ensure_dir(output_dir)
    outfilename = join(output_dir, 'fold_'+str(fold_index)+det_policy+'.csv')
    ##########################
    dataroot = get_ids18_root()
    benign_labels = pd.read_csv(join(dataroot,'category','benign_list.csv'),header=None)[0].values
    short_attack_labels = pd.read_csv(join(dataroot,'category','short_attack_list.csv'),header=None)[0].values
    long_attack_labels = pd.read_csv(join(dataroot,'category','long_attack_list.csv'),header=None)[0].values
    ##########################

    eps = 0.000000000000005
    df = pd.DataFrame(columns=['Label','Acc','Pr','Rc','F1-score'])

    acc = np.trace(cm)/np.sum(cm)

    for label in benign_labels:
        if label in cm_labels:
            i = np.where(label==cm_labels)[0][0]
            tp = cm[i,i]
            fp = np.sum(cm[:,i]) - cm[i,i]
            fn = np.sum(cm[i,:]) - cm[i,i]
            #print("{:25} - > (tp,fp,fn)=({:10.0f},{:10.0f},{:10.0f})".format(label,tp,fp,fn))
            pr = tp/(fp+tp+eps)
            rc = tp/(fn+tp+eps)
            f1 = 2*pr*rc/(pr+rc+eps)
        else:
            pr = 0
            rc = 0
            f1 = 0
        df = df.append({'Label':label,'Acc':acc,'Pr':pr,'Rc':rc,'F1-score':f1},ignore_index=True)


    total_n = np.sum(cm)
    m_f1=0
    m_pr = 0
    m_rc = 0

    w_f1=0
    w_pr = 0
    w_rc = 0

    n = len(cm_labels)-1# exclude benign
    for label in short_attack_labels:
        if label in cm_labels:
            i = np.where(label==cm_labels)[0][0]
            tp = cm[i,i]
            fp = np.sum(cm[:,i]) - cm[i,i]
            fn = np.sum(cm[i,:]) - cm[i,i]
            #print("{:25} - > (tp,fp,fn)=({:10.0f},{:10.0f},{:10.0f})".format(label,tp,fp,fn))
            pr = tp/(fp+tp+eps)
            rc = tp/(fn+tp+eps)
            f1 = 2*pr*rc/(pr+rc+eps)
        else:
            pr = 0
            rc = 0
            f1 = 0

        df = df.append({'Label':label,'Acc':acc,'Pr':pr,'Rc':rc,'F1-score':f1},ignore_index=True)
        m_pr+=pr*1./n
        m_rc += rc*1./n
        m_f1 +=f1*1./n

        w_pr+=pr  * np.sum(cm[i,:])/total_n
        w_rc += rc* np.sum(cm[i,:])/total_n
        w_f1 +=f1 * np.sum(cm[i,:])/total_n

    for label in long_attack_labels:
        if label in cm_labels:
            i = np.where(label==cm_labels)[0][0]
            tp = cm[i,i]
            fp = np.sum(cm[:,i]) - cm[i,i]
            fn = np.sum(cm[i,:]) - cm[i,i]
            #print("{:25} - > (tp,fp,fn)=({:10.0f},{:10.0f},{:10.0f})".format(label,tp,fp,fn))
            pr = tp/(fp+tp+eps)
            rc = tp/(fn+tp+eps)
            f1 = 2*pr*rc/(pr+rc+eps)
        else:
            pr = 0
            rc = 0
            f1 = 0

        df = df.append({'Label':label,'Acc':acc,'Pr':pr,'Rc':rc,'F1-score':f1},ignore_index=True)

        m_pr+=pr*1./n
        m_rc += rc*1./n
        m_f1 +=f1*1./n

        w_pr+=pr  * np.sum(cm[i,:])/total_n
        w_rc += rc* np.sum(cm[i,:])/total_n
        w_f1 +=f1 * np.sum(cm[i,:])/total_n

    df = df.append({'Label':'Macro Average of Attacks','Pr':m_pr,'Rc':m_rc,'F1-score':m_f1},ignore_index=True)
    df = df.append({'Label':'Weighted Average of Attacks','Pr':w_pr,'Rc':w_rc,'F1-score':w_f1},ignore_index=True)

    columns = ['Acc','Pr','Rc','F1-score']
    for col in columns:
        df[col] = 100*df[col]#convert to %
        df[col] = df[col].round(2)
    df.to_csv(outfilename)


def print_evaluation_ddos19(cm,cm_labels,output_dir):
    ensure_dir(output_dir)
    cm_labels = np.array(cm_labels)

    benign_labels = pd.read_csv(join(get_ddos19_root(),'benign_list.csv'),header=None)[0].values
    attack_labels = pd.read_csv(join(get_ddos19_root(),'attack_list.csv'),header=None)[0].values

    df = pd.DataFrame(columns=['Label','Acc','Pr','Rc','F1-score'])

    acc = np.trace(cm)/np.sum(cm)
    eps = 0.000000000000005

    for label in benign_labels:
        if label in cm_labels:
            i = np.where(label==cm_labels)[0][0]
            tp = cm[i,i]
            fp = np.sum(cm[:,i]) - cm[i,i]
            fn = np.sum(cm[i,:]) - cm[i,i]
            #print("{:25} - > (tp,fp,fn)=({:10.0f},{:10.0f},{:10.0f})".format(label,tp,fp,fn))
            pr = tp/(fp+tp+eps)
            rc = tp/(fn+tp+eps)
            f1 = 2*pr*rc/(pr+rc+eps)
        else:
            pr = 0
            rc = 0
            f1 = 0
        df = df.append({'Label':label,'Acc':acc,'Pr':pr,'Rc':rc,'F1-score':f1},ignore_index=True)

    total_n = np.sum(cm)
    m_f1=0
    m_pr = 0
    m_rc = 0

    w_f1=0
    w_pr = 0
    w_rc = 0

    n = len(cm_labels)-1# exclude benign
    for label in attack_labels:
        if label in cm_labels:
            i = np.where(label==cm_labels)[0][0]
            tp = cm[i,i]
            fp = np.sum(cm[:,i]) - cm[i,i]
            fn = np.sum(cm[i,:]) - cm[i,i]
            #print("{:25} - > (tp,fp,fn)=({:10.0f},{:10.0f},{:10.0f})".format(label,tp,fp,fn))
            pr = tp/(fp+tp+eps)
            rc = tp/(fn+tp+eps)
            f1 = 2*pr*rc/(pr+rc+eps)
        else:
            pr = 0
            rc = 0
            f1 = 0

        df = df.append({'Label':label,'Acc':acc,'Pr':pr,'Rc':rc,'F1-score':f1},ignore_index=True)
        m_pr+=pr*1./n
        m_rc += rc*1./n
        m_f1 +=f1*1./n

        w_pr+=pr  * np.sum(cm[i,:])/total_n
        w_rc += rc* np.sum(cm[i,:])/total_n
        w_f1 +=f1 * np.sum(cm[i,:])/total_n

    df = df.append({'Label':'Macro Average of Attacks','Pr':m_pr,'Rc':m_rc,'F1-score':m_f1},ignore_index=True)
    df = df.append({'Label':'Weighted Average of Attacks','Pr':w_pr,'Rc':w_rc,'F1-score':w_f1},ignore_index=True)

    columns = ['Acc','Pr','Rc','F1-score']
    for col in columns:
        df[col] = 100*df[col]#convert to %
        df[col] = df[col].round(2)
    df.to_csv(join(output_dir,'NIDS-performance.csv'))


def print_absolute_recall_ddos19(cm,cm_labels,output_dir):
    ensure_dir(output_dir)

    #####################
    gt_df = pd.read_csv(join(get_ddos19_root(),'CSVs/WS/PCAP-03-11_l/flow_dist.csv'),encoding='utf-8',usecols=['Label','Count'],dtype={'Label':str,'Count':int})
    all_label_flow_count = gt_df['Count'].sum()
    benign_labels = pd.read_csv(join(get_ddos19_root(),'benign_list.csv'),header=None)[0].values
    attack_labels = pd.read_csv(join(get_ddos19_root(),'attack_list.csv'),header=None)[0].values
    ##########################

    #------------------------#
    df = pd.DataFrame(columns=['Label','Rc'])
    #------------------------#

    m_rc = 0
    w_rc = 0
    for label in benign_labels:
        gt_count = gt_df[gt_df['Label']==label]['Count'].values[0]
        if label in cm_labels:
            index = np.where(label==cm_labels)[0][0]
            tp = cm[index,index]
        else:
            tp = 0
        test_set_count = gt_count
        rc = 100*tp/test_set_count # recall in %
        df = df.append({'Label':label,'Rc':rc}, ignore_index=True)
    
    num_mal_categories = gt_df.shape[0]-1# exclude benign
    for label in attack_labels:
        gt_count = gt_df[gt_df['Label']==label]['Count'].values[0]
        if label in cm_labels:
            index = np.where(label==cm_labels)[0][0]
            tp = cm[index,index]
        else:
            tp = 0
        test_set_count = gt_count
        rc = 100*tp/test_set_count # recall in %
        m_rc += rc*(1./num_mal_categories)
        w_rc += rc*(test_set_count/all_label_flow_count)
        df = df.append({'Label':label,'Rc':rc}, ignore_index=True)

    df = df.append({'Label':'Macro Average of Attacks','Rc':m_rc},ignore_index=True)
    df = df.append({'Label':'Weighted Average of Attacks','Rc':w_rc},ignore_index=True)

    df['Rc'] = df['Rc'].round(2)
    df.to_csv(join(output_dir,'detection_rate.csv'),index=False)


def print_absolute_recall(cm,cm_labels,output_dir,fold_index,det_policy):
    cm_labels = np.array(cm_labels)
    ensure_dir(output_dir)
    outfilename = join(output_dir, 'fold_'+str(fold_index)+det_policy+'.csv')

    if 'avg' in fold_index:
        K=1    
    else:
        K=5
    #####################
    dataroot = get_ids18_root()
    gt_df = pd.read_csv(join(dataroot,'CSVs/WS_l/flow_dist.csv'),encoding='utf-8',usecols=['Label','Count'],dtype={'Label':str,'Count':int})
    mal_flow_count = gt_df['Count'].sum()-gt_df[gt_df['Label']=='Benign']['Count'].values[0]

    benign_labels = pd.read_csv(join(dataroot,'benign_list.csv'),header=None)[0].values
    short_attack_labels = pd.read_csv(join(dataroot,'short_attack_list.csv'),header=None)[0].values
    long_attack_labels = pd.read_csv(join(dataroot,'long_attack_list.csv'),header=None)[0].values
    ##########################

    #------------------------#
    df = pd.DataFrame(columns=['Label','Rc'])
    #------------------------#
    for label in benign_labels:
        if label in cm_labels:

            res = np.where(label==cm_labels)
            index = np.where(label==cm_labels)[0][0]
            tp = cm[index,index]
        else:
            tp = 0
        test_set_count = gt_df[gt_df['Label']==label]['Count'].values[0]
        rc = 100*tp/(test_set_count/K) # recall in %
        df = df.append({'Label':label,'Rc':rc}, ignore_index=True)

    weight_sum=0
    m_rc = 0
    w_rc = 0

    num_mal_categories = gt_df.shape[0]-1# exclude benign
    for label in short_attack_labels:
        if label in cm_labels:
            index = np.where(label==cm_labels)[0][0]
            tp = cm[index,index]
        else:
            tp = 0
        gt_count = gt_df[gt_df['Label']==label]['Count'].values[0]
        rc = 100*tp/(gt_count/K) # recall in %
        m_rc += rc*(1./num_mal_categories)
        weight = gt_count/mal_flow_count
        w_rc += rc*weight
        df = df.append({'Label':label,'Rc':rc}, ignore_index=True)


    for label in long_attack_labels:
        if label in cm_labels:
            index = np.where(label==cm_labels)[0][0]
            tp = cm[index,index]
        else:
            tp = 0
        gt_count = gt_df[gt_df['Label']==label]['Count'].values[0]
        rc = 100*tp/(gt_count/K) # recall in %
        m_rc += rc*(1./num_mal_categories)
        weight = gt_count/mal_flow_count
        w_rc += rc*weight
        df = df.append({'Label':label,'Rc':rc}, ignore_index=True)
    df = df.append({'Label':'Macro Average of Attacks','Rc':m_rc},ignore_index=True)
    df = df.append({'Label':'Weighted Average of Attacks','Rc':w_rc},ignore_index=True)

    df['Rc'] = df['Rc'].round(2)
    df.to_csv(outfilename,index=False)


def get_cm(y_pred,y_true,test_id,labels):
    class_idx = np.sort(np.unique(y_true))
    num_class = len(class_idx)
    cm = np.zeros((num_class,num_class),dtype=int)

    for row_id in class_idx:

        indices = np.where(y_true==row_id)
        pred_r = y_pred[indices]
        test_id_r = test_id[indices]
        tp_indices = np.where(pred_r==row_id)
        tp_test_id_r = test_id_r[tp_indices]
        tp_ids = set(tp_test_id_r)
        previous_ids =set([])
        row = []
        for col_id in class_idx:
            if col_id==row_id:
                cm[row_id,col_id] = len(tp_ids)
            else:
                c_indices = np.where(pred_r==col_id)
                ids = set(test_id_r[c_indices])
                ids_after = (ids.difference(tp_ids)).difference(previous_ids)
                cm[row_id,col_id] = len(ids)
                previous_ids= previous_ids.union(ids_after)
    return cm


def result_logger_ids18(fingerprint, y_test, cm_tuple,fold_index):
    cmdir = join(fingerprint,'cm')
    recalldir = join(fingerprint,'recall')
    evaldir = join(fingerprint,'eval')
    ensure_dirs([cmdir,recalldir,evaldir])
    (cm_any, cm_majority, cm_all) = cm_tuple
    
    cm_ids = np.unique(y_test)
    _, id_to_label, _ = get_ids18_mappers()
    cm_labels = np.array([id_to_label[cm_id] for cm_id in cm_ids])
    #logging, plotting
    
    plot_confusion_matrix(join(cmdir,'any_{}.jpg'.format(fold_index)), [], [],cm=cm_any, classes=cm_labels, id_to_label=id_to_label)
    plot_confusion_matrix(join(cmdir,'majority_{}.jpg'.format(fold_index)), [], [],cm=cm_majority, classes=cm_labels, id_to_label=id_to_label)
    plot_confusion_matrix(join(cmdir,'all_{}.jpg'.format(fold_index)), [], [],cm=cm_all, classes=cm_labels, id_to_label=id_to_label)

    plot_confusion_matrix(join(cmdir,'any_norm_{}.jpg'.format(fold_index)), [], [],cm=cm_any, classes=cm_labels,id_to_label=id_to_label, normalize=True)
    plot_confusion_matrix(join(cmdir,'majority_norm_{}.jpg'.format(fold_index)), [], [],cm=cm_majority, classes=cm_labels,id_to_label=id_to_label, normalize=True)
    plot_confusion_matrix(join(cmdir,'all_norm_{}.jpg'.format(fold_index)), [], [],cm=cm_all, classes=cm_labels,id_to_label=id_to_label, normalize=True)


    print_evaluation(cm_any, cm_labels, evaldir, fold_index, 'any')
    print_evaluation(cm_majority, cm_labels, evaldir, fold_index, 'majority')
    print_evaluation(cm_all, cm_labels, evaldir, fold_index, 'all')

    print_absolute_recall(cm_any, cm_labels, recalldir, fold_index, 'any')
    print_absolute_recall(cm_majority, cm_labels, recalldir, fold_index, 'majority')
    print_absolute_recall(cm_all, cm_labels, recalldir, fold_index, 'all')


def result_logger_ddos19(fingerprint,y_test_perflowid_list,pred_list_tuples,id_to_label):
    cmdir = join(fingerprint,'cm')
    recalldir = join(fingerprint,'recall')
    evaldir = join(fingerprint,'eval')
    ensure_dirs([cmdir,recalldir,evaldir])

    (pred_any_list, pred_majority_list, pred_all_list) = pred_list_tuples

    # cm
    cm_any = confusion_matrix(np.array(y_test_perflowid_list),np.array(pred_any_list))
    cm_majority = confusion_matrix(np.array(y_test_perflowid_list),np.array(pred_majority_list))
    cm_all = confusion_matrix(np.array(y_test_perflowid_list),np.array(pred_all_list))

    cm_ids = np.unique(y_test_perflowid_list)
    cm_labels = np.array([id_to_label[cm_id] for cm_id in cm_ids])

    #logging, plotting
    plot_confusion_matrix(join(cmdir,'any_norm.jpg'), [], [],cm=cm_any, classes=cm_labels, normalize=True)
    plot_confusion_matrix(join(cmdir,'majority_norm.jpg'), [], [],cm=cm_majority, classes=cm_labels, normalize=True)
    plot_confusion_matrix(join(cmdir,'all_norm.jpg'), [], [],cm=cm_all, classes=cm_labels, normalize=True)

    print_evaluation_ddos19(cm_any, cm_labels, join(evaldir,'any'))
    print_evaluation_ddos19(cm_majority, cm_labels, join(evaldir,'majority'))
    print_evaluation_ddos19(cm_all, cm_labels, join(evaldir,'all'))

    print_absolute_recall_ddos19(cm_any, cm_labels, join(recalldir,'any'))
    print_absolute_recall_ddos19(cm_majority, cm_labels, join(recalldir,'majority'))
    print_absolute_recall_ddos19(cm_all, cm_labels, join(recalldir,'all'))
