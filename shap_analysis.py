import shap
import matplotlib.pyplot as plt
from utils import load_model_n_data
import numpy as np
import pandas as pd
from os.path import join


def plot_graph(models, X_tests, d, label):
        # region plot shap
        shap_values = []
        test_features = []
   
        for model, X_test in zip(models, X_tests):
            print(X_test.shape)
            explainer = shap.TreeExplainer(model)
            shap_value = explainer.shap_values(X_test)
            print("#classses = ", model.classes_ ) 
            fig = plt.figure()
            plt.rcParams['axes.facecolor'] = '#f2f8ff'
            print('shap_value.shape', len(shap_value), shap_value[0].shape)
            shap.summary_plot(shap_value[0], X_test, cmap=plt.get_cmap('gray'), sort=False, class_names=model.classes_,  max_display=X_test.shape[1])
            fig.tight_layout()
            
            plt.savefig(join(d,'{}.png'.format(label)))
            for 
            plt.show()

if __name__=='__main__':
    d= '/data/juma/data/ids18/CSVs_r_0.001_m_1.0/SI_100/SFS_SI_95.33_l'
    labels = ['Brute Force-Web', 'Brute Force-XSS']
    for label in labels[:1]:
        models, X_tests = load_model_n_data(d, label)
        plot_graph(models, X_tests, d, label)
