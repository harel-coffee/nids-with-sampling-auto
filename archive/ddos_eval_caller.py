from ddos_evaluator import evaluate

train_fingerprint = 'RPS_SI_10'
test_fingerprint =  'RPS_SI_10'
traindir = '/data/juma/data/ddos/CSVs_r_1.0/SR_10/{}/PCAP-01-12_l'.format(train_fingerprint) 

for i in range(6,7):
    ratio = 1./10**i
    csv_dir = 'CSVs_r_{}'.format(ratio)
    testdir = traindir.replace('CSVs_r_1.0',csv_dir).replace('PCAP-01-12_l','PCAP-03-11_l').replace(train_fingerprint,test_fingerprint)
    print("Evaluating .. {}\r".format(testdir))
    evaluate(traindir,testdir,'cnn')    
    
