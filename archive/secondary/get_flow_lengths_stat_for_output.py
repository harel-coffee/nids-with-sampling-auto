import numpy as np
from collections import defaultdict
import os
from os.path import join as os_join
from utils import get_complete_label_names,read_csv
from tqdm import tqdm


dataroot = '/home/isrl/data/CIC-IDS-2017/PCAPs/output'
labelled_dataroot = '/home/isrl/data/CIC-IDS-2017/GeneratedLabelledFlows/TrafficLabellingMerged'
good_flowid_root = '/home/isrl/data/CIC-IDS-2017/PCAPs/output'
EST = True

def make_dictionary_from_flow(data):
    d = defaultdict(list)
    print('making dictionary from flow')
    for row in tqdm(data):
        flowid = row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]+'-'+row[5]  
        d[flowid].append(row) 
    return d


def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir,name)) and not name.startswith(".~lock.") and (name.endswith(".pcap_ISCX.csv") or name.endswith(".pcap_Flow.csv"))]


label_names = get_complete_label_names()

if EST:
    output_filename = os_join(dataroot,"percentiles_for_num_of_packets_est.csv")
else:
    output_filename = os_join(dataroot,"percentiles_for_num_of_packets_bidirection.csv")    
if os.path.exists(output_filename):
    os.remove(output_filename)

filenames = get_filenames(dataroot)
bidirectional_flow_lengths = defaultdict(list)
backward_flow_lengths = defaultdict(list)
forward_flow_lengths = defaultdict(list)

for filename in filenames: # E: Friday-WorkingHours.pcap_Flow.csv
    data = read_csv(os_join(dataroot,filename))
    data_d = make_dictionary_from_flow(data)
   
    labelled_data = read_csv(os_join(labelled_dataroot,filename))
    labelled_data_d = make_dictionary_from_flow(labelled_data)
 
    too_many_labels = 0
    no_labelled_data = 0
    for i in range(data.shape[0]):
        row = data[i]
        flowid = row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]+'-'+row[5]
        data_flow = row
 
        labeled_flow = labelled_data_d[flowid]
        labeled_flow = np.array(labeled_flow)
        
        if labeled_flow.shape[0]<1:
            no_labelled_data +=1
            continue
            print('No labelled data for corresponding flow={:40}. \n This should not happen. Please investigate code'.format(flowid))
            exit(1)

        if len(set(labeled_flow[:,-1]))>1:
            too_many_labels+=1
            continue

        flow_label = labeled_flow[0][-1]
        if EST:
            fwd_pkts = int(row[8])
            bwd_pkts = int(row[9])
            bidir_packets = fwd_pkts+bwd_pkts
            
        else:
            forward_packets = int(labeled_flow[0][8])
        
        bidirectional_flow_lengths[flow_label].append(bidir_packets)
        forward_flow_lengths[flow_label].append(fwd_pkts)
        backward_flow_lengths[flow_label].append(bwd_pkts)

    print('Too many labels: ', too_many_labels)
    print('no_labelled_data: ', no_labelled_data)

tpl = [(bidirectional_flow_lengths,'_bidir.csv'),(forward_flow_lengths,'_fwd.csv'),(backward_flow_lengths,'_bwd.csv')]
for flow_lengths,output_ext in tpl:
    percentile = np.array(['Label','num_flows','Min','20-th','50-th','90-th percentile','95-th percentile','99-th percentile','99.9-th percentile','100-th percentile'])
    percentile = percentile.reshape((1,-1))

    for label in label_names:  
        flow_lengths_for_label = flow_lengths[label]
        print("{:40s}-->{:10d}".format(label,len(flow_lengths_for_label)))

        if len(flow_lengths_for_label)<1:
            continue
        flow_lengths_for_label = np.array(flow_lengths_for_label)
  
        row = np.array([label,len(flow_lengths_for_label),np.min(flow_lengths_for_label),\
           np.percentile(flow_lengths_for_label,20),np.percentile(flow_lengths_for_label,50),\
           np.percentile(flow_lengths_for_label,90),np.percentile(flow_lengths_for_label,95),np.percentile(flow_lengths_for_label,99),\
           np.percentile(flow_lengths_for_label,99.9), np.percentile(flow_lengths_for_label,100)])
        percentile = np.concatenate((percentile,row.reshape((1,-1))),axis=0)
        np.savetxt(output_filename.replace('.csv',output_ext),percentile,fmt='%s',delimiter=',')
        
