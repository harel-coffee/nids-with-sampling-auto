import glob
from kfold_classifier import classify
from os.path import join


classifier_name = 'tree'
#classifier_name = 'forest'
#classifier_name = 'softmax'
#classifier_name='cnn'

root = '/data/juma/data/ids18/CSVs_r_1.0/SR_10/'
for i,d in enumerate(glob.glob(join(root,"*_l"))):
    print(i,d,classifier_name)
    classify(d,classifier_name)
