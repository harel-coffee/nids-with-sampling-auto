import pandas as pd
#fn = '/data/juma/data/ids18/CSVs_before_split_fix/CSVs/WS_l/reduced_fold_p20_2.csv'
#fn = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/WS_l/r_fold_2.csv'
fn = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/WS_l/fold_2.csv'

df = pd.read_csv(fn, usecols=['Label'])
print(df.Label.value_counts())
