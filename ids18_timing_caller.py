import glob
from timing_inference import classify
from os.path import join

root = '/data/juma/data/ids18/CSVs_r_0.001/SR_10/'
file_ending =  '*_CICFlowMeter.csv'


#classifier_name = 'tree'
#classifier_name = 'forest'
#classifier_name = 'softmax'
classifier_name='cnn'

for i,d in enumerate(glob.glob(join(root,"*_l"))):
    print('\n----------------\n',i,d,classifier_name)
    classify(d,classifier_name)

