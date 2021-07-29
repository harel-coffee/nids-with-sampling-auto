import os
import glob
import pandas as pd
from os.path import join
dataroot = '/hdd/juma/data/net_intrusion/CIC-IDS-2018/CSVs/sk_sr_1.0'


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and not 'archive' in name]


def merge(dataroot)
	folders = get_immediate_subdirectories(dataroot)
	print(folders)
	for folder in folders:
    	filenames = [i for i in glob.glob(join(dataroot,folder,'*.pcap_Flow.csv'))]
    	combined_csv = pd.concat([pd.read_csv(f) for f in filenames],sort=False)
    	print(folder,combined_csv.columns.values)
    	print()
    	combined_csv.to_csv(join(dataroot,folder+'_TrafficForML_CICFlowMeter.csv'),index=False,encoding='utf-8-sig')


def label_flows (data, attack_source, attack_time, attack_names, save_path, infiltration=False):
    data = np.array(data)
    #not_flipped=True
    for ttx, attack_name in enumerate(attack_names):
        for source in attack_source[ttx]:
            for idx, record in enumerate(data):
                if idx == 0:
                    continue
                else:
                    #if ttx == 0 and not_flipped:
                    #    data[idx][0]= order_flowid(record[0])
                    if infiltration:
                        if record[3] == source and record[6]>=attack_time[ttx][0] and record[6]<=attack_time[ttx][1]:
                            data[idx][-1]= attack_name+'\n'
                        else:
                            continue
                    else:
                        if record[1] == source and record[6]>=attack_time[ttx][0] and record[6]<=attack_time[ttx][1]:
                            data[idx][-1]= attack_name+'\n'
                        else:
                            continue
            #not_flipped=False
    
    data = clear_labels(data)
    if save_path != False:
        np.savetxt(save_path, data, delimiter=",", fmt='%s')
    return data        


merge(dataroot)

days = [
'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',
'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',
'Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv',
'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv',
'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv'
]

schedule = []

for csv_file in glob.glob(join(dataroot,'*_CICFlowMeter.csv')):
	data = pd.read_csv(csv_file)
	data = label_flows(data,)
 

