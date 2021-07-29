mport os
import csv
from utils import read_csv,make_dictionary_from_flowi


dataroot = ''
def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir,name)) and not name.startswith(".~lock.") and (name.endswith(".pcap_ISCX.csv") or name.endswith(".pcap_Flow.csv"))]



filenames = get_filenames()
with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    line = -1
    for row in csv_reader:
        line+=1
        if line==0:
            print(row)
            continue
        else:
            print(row)
            break
        line+=1

data = read_csv(filename)
print(data.shape)
data_d = make_dictionary_from_flow(data)

