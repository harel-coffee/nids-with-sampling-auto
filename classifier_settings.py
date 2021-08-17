from os.path import join

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from models import Classifier
import pickle
from utils import get_ids18_mappers

SEED = 234
class ClassifierLoader:
    def __init__(self):
        pass

    def load(self,args):
        classifier_name = args['classifier_name']
        if classifier_name=='tree':
            return self.load_tree(args)
        elif classifier_name=='forest':
            return self.load_forest(args)
        elif classifier_name=='softmax':
            return self.load_softmax(args)
        elif classifier_name=='cnn':
            return self.load_cnn(args)
        else:
            print("No such classifier as {}".format(classifier_name))

    def load_tree(self,args):
        balance = args['balance']
        model_file = open(join(args['runs_dir'],'model.pkl'),'rb')
        if balance=='with_loss':
            return pickle.load(model_file)
        elif balance=='explicit' or balance=='no':
            return pickle.load(model_file)
        elif balance =='sample_per_batch':
            print("Decision Tree does not use mini batch")
            exit()

    def load_forest(self,args):
        balance = args['balance']
        bootstrap = args['bootstrap']
        model_file = open(join(args['runs_dir'],'model.pkl'),'rb')
        if balance=='with_loss':
            return pickle.load(model_file)
        elif balance=='explicit' or balance=='no':
            return pickle.load(model_file)
        elif balance=='sample_per_batch':
            print("Forst does not use mini batch")
            exit()

    def load_softmax(self,args):
        clf = Classifier(args,method='softmax')
        return clf

    def load_cnn(self,args):
        clf = Classifier(args,method='cnn2')
        
        return clf


def get_num_ws_classes():
    label_to_id, id_to_label, _ = get_ids18_mappers()
    return len(label_to_id)


def get_classifier_dir(sampler_dir, classifier_name, class_weight=None):
    _, config = get_args(classifier_name, class_weight)
    clf_dir = join(sampler_dir, 'c_{}'.format(classifier_name)+config)
    return clf_dir


def load_classifier(classifier_name,model_file,  class_weight=None):
    args, _  = get_args(classifier_name, class_weight)
    args['runs_dir'] = model_file
    CL = ClassifierLoader()
    return CL.load(args) 


def get_args(classifier_name, class_weight):
        input_dim = 78 # because we remove Label,FlowID,Timestamp columns from X
        #note, num_class is only used by CNN
        balance = get_balancing_technique()
        if classifier_name in ['cnn','softmax']:
            batch_size = 4096 
            optim='Adam'
            device =0
            
            if classifier_name=='cnn':
                lr =1e-3
                reg = 0
            elif classifier_name=='softmax':
                lr = 1e-3
                reg =0
            classifier_args = {'classifier_name':classifier_name,'optim':optim,'lr':lr,'reg':reg,'batch_size':batch_size,'input_dim':input_dim,'num_class':get_num_ws_classes(),'device':device, 'balance':balance, 'class_weight':class_weight}
            config =  '_optim_{}_lr_{}_reg_{}_bs_{}_b_{}'.format(optim,lr,reg,batch_size,balance)
        else:
            if classifier_name=='forest':
                bootstrap = True
                max_features = 'auto'
                #max_features = None
                n_estimators = 100
                min_samples_leaf = 3 # related to major/all/any supposed to increase all and speed up building
                max_samples = .01
                max_depth = 25
                classifier_args = {'classifier_name':classifier_name,'balance':balance,'bootstrap':bootstrap,'max_features':max_features, 'n_estimators':n_estimators, 'min_samples_leaf':min_samples_leaf, 'max_samples':max_samples, 'max_depth':max_depth}
                config = '_b_{}_n_{}_bootstrap_{}_mf_{}_msl_{}_ms_{}_md_{}'.format(balance,n_estimators,bootstrap,max_features, min_samples_leaf, max_samples, max_depth)

            elif classifier_name == 'tree':
                max_features = 'auto'
                min_samples_leaf = 3
                max_depth = 25
                classifier_args = {'classifier_name':classifier_name,'balance':balance, 'max_features':max_features, 'min_samples_leaf':min_samples_leaf, 'max_depth':max_depth}
                config = '_b_{}_mf_{}_msl_{}_md_{}'.format(balance, max_features, min_samples_leaf, max_depth)
        return classifier_args, config


def get_classifier(args):
    classifier_name = args['classifier_name']
    balance = args['balance']

    if classifier_name=='tree':
        if balance=='with_loss' :
            return tree.DecisionTreeClassifier(class_weight='balanced',max_features=args['max_features'],min_samples_leaf=args['min_samples_leaf'])
        elif balance=='explicit' or balance=='no':
            return tree.DecisionTreeClassifier(max_features=args['max_features'], min_samples_leaf=args['min_samples_leaf'])

    elif classifier_name=='forest':
        bootstrap = args['bootstrap']
        if balance=='with_loss':
            return RandomForestClassifier(n_estimators=args['n_estimators'],class_weight='balanced',n_jobs=-1,random_state=SEED, bootstrap=bootstrap,max_features=args['max_features'], min_samples_leaf=args['min_samples_leaf'],max_samples=args['max_samples'],max_depth=args['max_depth'])
        elif balance=='with_loss_sub':
            return RandomForestClassifier(n_estimators=args['n_estimators'],class_weight='balanced_subsample',n_jobs=-1,random_state=SEED, bootstrap=bootstrap,max_features=args['max_features'], min_samples_leaf=args['min_samples_leaf'],max_samples=args['max_samples'],max_depth=args['max_depth'])

        elif balance=='explicit' or balance=='no':
            return RandomForestClassifier(n_estimators=args['n_estimators'],n_jobs=None,random_state=SEED, bootstrap=bootstrap, max_features = args['max_features'], min_samples_leaf = args['min_samples_leaf'], max_samples=args['max_samples'], max_depth=args['max_depth'])

    elif classifier_name=='softmax':
        clf = Classifier(args,method='softmax')
        return clf

    elif classifier_name=='cnn':
        clf = Classifier(args,method='cnn2')
        return clf

def get_balancing_technique():
        #balance = 'no'
        #balance = 'with_loss'
        #balance = 'with_loss_inverse'
        #balance = 'with_loss_sub'
        balance = 'explicit'
        #balance = 'sample_per_batch'
        return balance
