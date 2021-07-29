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

    def __init__(self,csv_file, chunksize=10**6):
        self.csv_file = csv_file
        self.num_records = self.get_num_records(csv_file)
        self.chunksize = chunksize
        self.seen_so_far = 0 # number of flow records seen so far
        self.seen_chunks = 0
        self.iterableReader = pd.read_csv(csv_file, engine='c', usecols=get_cols4ml(), dtype=get_dtype4normalized(),  chunksize=chunksize)

        label_to_id, id_to_label, _ = get_ids18_mappers() 
        self.label_to_id = label_to_id                


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
        if self.seen_so_far%self.chunksize==0:
            self.seen_chunks=self.seen_so_far//self.chunksize
            self.load_chunk()
         
        self.seen_so_far+=1
        #cnk_idx = idx%self.y.shape[0]
        cnk_idx = idx - self.seen_chunks*self.chunksize
        try:
            res = (self.x[cnk_idx], self.y[cnk_idx])
        except:
            print(idx,self.seen_chunks)
            print("No such index exists",self.x.shape,cnk_idx, idx)
            exit()
        return res
    
    def get_num_records(self,csv_file):
        result = subprocess.run(['wc','-l',csv_file], stdout=subprocess.PIPE)
        num_records = int(result.stdout.split()[0])-2
        return 1000*num_records//1000

    
    def encode_label(self, str_labels):
        return [self.label_to_id[str_label] for str_label in str_labels]


    def load_chunk(self):
        if self.seen_so_far%self.chunksize!=0:
            print("Prev chunk is not finished cleanly ", self.seen_so_far%self.chunksize)
        
        seen_chunks_so_far = self.seen_so_far//self.chunksize
        df = next(self.iterableReader)
        print("Loading chunk of shape ", df.shape, 'for ', ntpath.basename(self.csv_file))
        #print('Dst Port values ', df['Dst Port'].unique()) 
        if df.shape[0]==0:
            print("Done Reading")
            exit()       
        #print(df.Label.value_counts())
        self.x = torch.FloatTensor(df.drop(columns=['Label']).values)
        self.y = torch.LongTensor(self.encode_label(df.Label.values))
