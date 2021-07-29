#!/bin/bash
from subprocess import call, check_output, check_call
import os
from glob import glob
from os.path import join

dataroot = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/SR_0.1/'
#dataroot = '/data/juma/data/ids18/CSVs_r_1.0_m_1.0/'
#prefix = 'r'
prefix = 'bal'
for sdir in glob(join(dataroot,'*_l')):
    if 'SFS' not in sdir:
        continue
    os.chdir(sdir)
    print(os.getcwd())
    print('fold 2')
    check_output('cat  {}_fold_2.csv > {}_train.csv'.format(prefix, prefix), shell=True)
    print('fold 3')
    check_output('cat | tail -n+2 {}_fold_3.csv  >> {}_train.csv'.format(prefix, prefix), shell=True)
    print('fold 4')
    check_output('cat | tail -n+2 {}_fold_4.csv >> {}_train.csv'.format(prefix, prefix), shell=True)
    print('fold 1')
    check_output('cat | tail -n+400000 {}_fold_1.csv >> {}_train.csv'.format(prefix,prefix),shell=True)

