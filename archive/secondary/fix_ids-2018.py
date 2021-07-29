import glob
from os.path import join
import numpy as np

dataroot = '/home/isrl//data/net_intrusion/CIC-IDS-2018/CSVs/cicflowmeter_original/'

def fix_file(filename):
    contents = [i.split(',') for i in open(filename).readlines()]
    N = len(contents)
    print(N,len(contents[0]))
    contents = contents[slice(0,N,2)]
    contents = np.array(contents)
    print(contents.shape)
    contents = np.char.strip(contents,chars='"\'\"\n')
    #attacks = sorted(list(set(contents[:,-1])))
    #print(set(contents[:,-1]))
    #for i in range(contents.shape[0]):
    #    if contents[i,-1]=='DDoS-LOIC-UDP':
    #        print('match')
    contents = np.core.defchararray.replace(contents,'DDoS-LOIC-UDP','DDOS-LOIC-UDP')
    #contents = np.char.strip(contents,"\n")
    return contents

for i in range(1):
    filename = join(dataroot,'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv')
    print(filename)
    contents = fix_file(filename)
    #print(contents[:3])
    output_filename = filename.replace('cicflowmeter_original','cicflowmeter')
    np.savetxt(output_filename,contents,delimiter=',',fmt='%s')
