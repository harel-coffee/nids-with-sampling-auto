from glob import glob
from os.path import join
from make_fold import make_fold


sr_dataroot = '/data/juma/data/ids18/CSVs_r_0.4/SR_10/'
#sr_dataroot = '/mnt/sda_dir/juma/data/net_intrusion/CIC-IDS-2018/CSVs_r_0.2/SR_10/'
for i,d in enumerate(glob(join(sr_dataroot,'*_l'))):
    print(r'\n------{i}--------')
    print(d)
    make_fold(d)
