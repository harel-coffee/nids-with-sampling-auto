from os.path import join as os_join
from tqdm import tqdm
from collections import defaultdict
from numpy import genfromtxt
from utils import read_csv, read_csv_header
import numpy as np
import os
originroot =  '/home/isrl/data/CIC-IDS-2017/PCAPs/output_filtered/'
sFlow_root =  '/home/isrl/data/CIC-IDS-2017/PCAPs/sFlow_filtered/'
sketchflow_root = '/home/isrl/data/CIC-IDS-2017/PCAPs/sketchflow_filtered/'
threshold = 0 
output_root = originroot.replace('PCAPs/output_filtered/','comparison_{}'.format(threshold))
if not os.path.exists(output_root):
    os.mkdir(output_root)


def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir,name)) and not name.startswith(".~lock.") and (name.endswith(".pcap_ISCX.csv") or name.endswith(".pcap_Flow.csv"))]


def make_dictionary_from_flow(data):
    d = defaultdict(list)
    print('making dictionary from a flow')
    for row in tqdm(data):
        flowid = row[1]+'-'+row[2]+'-'+row[3]+'-'+row[4]+'-'+row[5]
        num_packets = min(float(row[7]),float(row[8]))
        if num_packets>=threshold:
            d[flowid].append(row) # row[0] is flow i

    d_filtered = defaultdict(list)
    for key,vals in d.items():
        if len(vals)<2:
            d_filtered[key] = vals

    return d_filtered


def get_flow_intersection(original_d,sketchflow_d,sFlow_d):
    # note, we are taking original data into account because small flows are filtered in it
    key0 = list(original_d.keys())
    key1 = list(sketchflow_d.keys())
    key2 = list(sFlow_d.keys())
    return set(key1).intersection(key2).intersection(key0)


summary_footer = []
sketchflow_error_mean_sum = 0.
feature_mean_sum = 0.
def calculate_est(filename):
    
    original_data = read_csv(os_join(originroot,filename))
    original_data_d = make_dictionary_from_flow(original_data)
    sFlow_data = read_csv(os_join(sFlow_root,filename))
    sFlow_data_d = make_dictionary_from_flow(sFlow_data)
    sketchflow_data = read_csv(os_join(sketchflow_root,filename))
    sketchflow_data_d = make_dictionary_from_flow(sketchflow_data)
    
    intersection = get_flow_intersection(original_data_d,sketchflow_data_d,sFlow_data_d)
    intersection = list(intersection)
   
    # filtering
    original_data =   np.array([original_data_d[key][0] for key in intersection if key in original_data_d ])
    sketchflow_data = np.array([sketchflow_data_d[key][0] for key in intersection if key in sketchflow_data_d])
    sFlow_data = np.array([sFlow_data_d[key][0] for key in intersection if key in sFlow_data_d])
    print(original_data.shape,sketchflow_data.shape,sFlow_data.shape,len(intersection))
    origin_header = read_csv_header(os_join(originroot,filename))
    flowid_indices = [0,1,2,3,4,5,6] # Flow identification, although i think we do not need timestamp
    num_ids = len(flowid_indices)
    num_features = original_data.shape[1] - num_ids
    N = original_data.shape[0] 
    estimation = np.empty([N,len(flowid_indices)+3*num_features],dtype=str)
    header = origin_header[:7]
    footer = ['Mean', 'of', 'abstract', 'error','for','each','feature']
    footer2 = ['Contains', 'errors', 'in', '%','','','']
    estimation[:,flowid_indices] = original_data[:,flowid_indices]
    for i in range(num_features):
        index_d = num_ids + i
        header.append('')
        header.append(origin_header[index_d] + ' - Sketchflow Est')
        header.append(origin_header[index_d] + ' - SFlow Est')
    estimation = original_data[:,flowid_indices[0]].reshape((-1,1)) #(N,1) column
    for index in flowid_indices[1:]:
        estimation = np.hstack((estimation,original_data[:,index].reshape((-1,1))))

    for i in range(num_features):
        index_d = num_ids + i
        index_e = num_ids + i*3
        
        estimation = np.hstack((estimation,original_data[:,index_d].reshape((-1,1))))
        estimation = np.hstack((estimation,sketchflow_data[:,index_d].reshape((-1,1))))
        estimation = np.hstack((estimation,sFlow_data[:,index_d].reshape((-1,1))))
        original_mean = np.mean(original_data[:,index_d].astype(float))
        sketchflow_mean_abs = np.mean(np.abs(original_data[:,index_d].astype(float)-sketchflow_data[:,index_d].astype(float)))
        sFlow_mean_abs = np.mean(np.abs(original_data[:,index_d].astype(float)-sFlow_data[:,index_d].astype(float)))
    #print("SK: {:.2f} | SF {:.2f}".format(sketchflow_mean_abs,sFlow_mean_abs))
        footer.append(original_mean)
        footer.append(sketchflow_mean_abs)
        footer.append(sFlow_mean_abs)
        #footer.append('')
       
        footer2.append(' ')
        #footer2.append("{:.2f}".format(original_mean))
        footer2.append("{:.2f}".format(100*sketchflow_mean_abs/(original_mean+0.00005)))
        footer2.append("{:.2f}".format(100*sFlow_mean_abs/(original_mean+0.000005)))
        
    header = np.array(header).reshape(1,-1)
    footer = np.array(footer).reshape(1,-1)
    footer2 = np.array(footer2).reshape(1,-1)
   
    estimation_data = np.concatenate((header,estimation,header,footer,footer2),axis=0)
    np.savetxt(os_join(output_root,filename),estimation_data,fmt='%s',delimiter=',')
    return header[:,7:],footer[:,7:],footer2[:,7:] # removing initials


filenames = get_filenames(originroot)

header, footer,footer2 = calculate_est(filenames[0])

header = header.reshape((-1,1))
#header_mean_error = make_mean_error_header(header)
#header_percentage_error = make_error_percentage(header)

footer2_week = footer2.reshape((-1,1))
footer_all = footer

vertical_header = ['']
i = filenames[0].find('-')
day = filenames[0][:i]
vertical_header.append(day)

for filename in filenames[1:]:
    _,footer,footer2 = calculate_est(filename)
    footer_all = np.concatenate((footer_all,footer),axis=0)
    footer2_week = np.hstack((footer2_week,footer2.reshape((-1,1))))
    i = filename.find('-')
    day = filename[:i]
    vertical_header.append(day)
vertical_header.append('Total')

footer_mean = np.mean(footer_all.astype('float64'),axis=0)
footer_percentage=np.empty(footer2_week.shape[0])
sketchflow_indices = np.arange(1,footer2_week.shape[0],3) 
sFlow_indices = np.arange(2,footer2_week.shape[0],3) 
eps = 0.0000005
footer_percentage[sketchflow_indices] = (footer_mean[sketchflow_indices]*100)/(footer_mean[sketchflow_indices-1] + eps)
footer_percentage[sFlow_indices] = (footer_mean[sFlow_indices]*100)/(footer_mean[sFlow_indices-2]+ eps)
footer_percentage = np.round(footer_percentage,2)
footer_percentage = footer_percentage.astype('str')
footer_percentage[sketchflow_indices-1]=''
print(header.shape,footer2_week.shape,footer_percentage.reshape((-1,1)).shape)

data = np.hstack((header,footer2_week,footer_percentage.reshape((-1,1))))    

vertical_header = np.array(vertical_header).reshape((1,-1)) 
data = np.concatenate((vertical_header,data),axis=0)
np.savetxt(os_join(output_root,'summary.csv'),data,fmt='%s',delimiter=',')

