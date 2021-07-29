import glob
from kfold_evaluator import classify
from os.path import join

file_ending =  '*_CICFlowMeter.csv'

#classifier_name = 'tree'
#classifier_name = 'forest'
#classifier_name = 'softmax'
classifier_name='cnn'
root_template = '/data/juma/data/ids18/CSVs_r_{}/SR_10/'

for i in range(2):
    ratio = 0.5/10**i
    print('\n*********************')
    print("ratio = ",ratio)
    root = root_template.format(ratio) 
    for j,d in enumerate(glob.glob(join(root,"*_l"))):
        print('\n----------------\n',j,d,classifier_name)
        classify(d,classifier_name)

