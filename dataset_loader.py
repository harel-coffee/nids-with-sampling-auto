from torch.utils.data import Dataset 
from utils import get_ids18_mappers
import pandas as pd
import torch 
from utils import getSeed
from utils import get_cols4ml, get_dtype4normalized
import subprocess
import ntpath


class FlowRecordDataset(Dataset):
    "Flow record dataset"

    def __init__(self,csv_file):
        self.csv_file = csv_file
        self.num_records = self.get_num_records(csv_file)
        
        df = pd.read_csv(csv_file, engine='c', usecols=get_cols4ml(), dtype=get_dtype4normalized())
        self.x = torch.FloatTensor(df.drop(columns=['Label']).values)
        self.y = torch.LongTensor(self.encode_label(df.Label.values))


    def __len__(self):
        return self.num_records

    
    def __getitem__(self, idx):
        #if self.seen_so_far==self.num_records:
        #    print("already seen all ",idx,self.seen_so_far, self.num_records, end=',')
        #    return 
#        print(idx,end=',')
        if torch.is_tensor(idx):
            print("__get_item__ is taking a list of idx. Code needs adjusting")
            idx = idx.tolist()
        res = (self.x[idx], self.y[idx])
        return res
    
    def get_num_records(self,csv_file):
        result = subprocess.run(['wc','-l',csv_file], stdout=subprocess.PIPE)
        num_records = int(result.stdout.split()[0])-1
        return num_records

    
    def encode_label(self, str_labels):
        label_to_id, id_to_label, _ = get_ids18_mappers() 
        return [label_to_id[str_label] for str_label in str_labels]


