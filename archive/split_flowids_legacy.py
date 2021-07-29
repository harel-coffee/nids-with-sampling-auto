import pandas as pd
from utils import getSeed
from glob import glob
from os.path import join
import ntpath
SEED = getSeed()


def read_flowIDs(dataroot,file_ending='*Meter.csv',fraction=1):
    filenames = [i for i in glob(join(dataroot,file_ending))]
    df_list = []
    for f in filenames:
        print("reading ", ntpath.basename(f))
        df = pd.read_csv(f,usecols=['Flow ID','Label'],dtype={'Flow ID':str,'Label':str})
        df_list.append(df.sample(frac=fraction,random_state=SEED))
    combined_csv= pd.concat(df_list,sort=False)
    return combined_csv



def group_data(df, K):
    #remove classes less than K items
    print("Grouping to remove small (er than K) classes")
    labels = [ label for (flowid,label) in df.groupby(['Flow ID','Label']).groups.keys()]
    unique,count = np.unique(labels,return_counts=True)
    print('-----------------------------------')
    print("REMOVING VERY SMALL CLASSES")
    for label,count in zip(unique,count):
        if count<K:
            df = df[df['Label']!=label]
            print(label)
    print('-----------------------------------')

    # after deleting small classes regroup again
    print("Re-Grouping by FlowID and Label")
    grouped = df.groupby(['Flow ID','Label'], sort=True)
    ID = [ [flowid,label]  for (flowid,label)  in grouped.groups.keys()]
    groupid,count = np.unique(ID,return_counts=True)

    Label = [label for flowid,label in ID]
    ID = np.array(ID)
    return ID,Label


if __name__=='__main__':

    dataroot = '/data/juma/data/ids18/CSVs/WS_l' 
    df = read_flowIDs(dataroot)

    flowids,flowlabels = group_data(df,K)
    unique_labels,label_counts = np.unique(flowlabels,return_counts=True)
    flow_observation_rate = np.ones(len(unique_labels))*100
    pd.DataFrame({'Label':unique_labels,'Count':label_counts,'Observation Rate':flow_observation_rate}).to_csv(join(outputdir,'flow_dist.csv'),index=False,encoding='utf-8-sig')

    skf = StratifiedKFold(n_splits=K,random_state=SEED)
    for fold_index, (train_index,test_index) in enumerate(skf.split(flowids,flowlabels)):
            print("Fold ",fold_index)
            print("Group IDs shape")
            print(train_index.shape,test_index.shape)
            tick = time.time()
            test_flowids = flowids[test_index]
            fold_df = get_flow_records(test_flowids,df)
            fold_df.to_csv(join(outputdir,'fold_{}.csv'.format(fold_index)),index=False, encoding='utf-8-sig')

