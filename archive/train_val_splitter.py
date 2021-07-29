#!/bin/bash
from subprocess import call, check_output, check_call
import os

dataroot = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/SR_1.0/FFS_(8,16,40)_l'
os.chdir(dataroot)
print(os.getcwd())
#call('cat <(cat n_bal_fold_2.csv) <(tail -n+2 n_bal_fold_3.csv) <(tail -n+2 n_bal_fold_4.csv) <(tail -n+400000 n_bal_fold_1.csv) > n_bal_train.csv', shell=True)

#os.system("cat <(cat bal_fold_2.csv) <'tail -n+2 bal_fold_3.csv' <'tail -n+2 bal_fold_4.csv' <'tail -n+400000 bal_fold_1.csv' > bal_train.csv", shell=True)
#call('cat < (cat bal_fold_2.csv) <(tail -n+2 bal_fold_3.csv) <(tail -n+2 bal_fold_4.csv) <(tail -n+400000 bal_fold_1.csv) > bal_train.csv', shell=False)

print('fold 2')
check_output('cat  bal_fold_2.csv > bal_train.csv', shell=True)
print('fold 3')
check_output('cat | tail -n+2 bal_fold_3.csv  >> bal_train.csv', shell=True)
print('fold 4')
check_output('cat | tail -n+2 bal_fold_4.csv >> bal_train.csv', shell=True)
print('fold 1')
check_output('cat | tail -n+400000 bal_fold_1.csv >> bal_train.csv',shell=True)

