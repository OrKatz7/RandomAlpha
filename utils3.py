import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from deepchecks.tabular.suites import full_suite
from sklearn.preprocessing import PowerTransformer,MinMaxScaler,QuantileTransformer
from deepchecks.tabular import Dataset
import time
import dabl
from tqdm.auto import tqdm
from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_evaluations, plot_convergence, plot_regret
from skopt.space import Categorical, Integer, Real
import random
from sklearn.base import is_classifier, is_regressor
from sklearn.tree import _tree
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from math import sqrt
from joblib import Parallel, delayed
from models_3 import DecisionTreeRegressor_KatzArbili,DecisionTreeClassifier_KatzArbili
from skopt import dump, load

def log(text,name):
    with open(f'log/{name}.log', 'a+') as logger:
        logger.write(f'{text}\n')
        
def calculate_final_score_class(rf_clf,x_val,y_val,A,N,T,CFG):
    y_pred = rf_clf.predict_proba(x_val,A,N,T,False)
    auc = roc_auc_score(y_val,y_pred if CFG.multi_class else y_pred[:,1],multi_class='ovo' if CFG.multi_class else "raise")
    return auc

def calculate_final_score_reg(rf_clf,x_val,y_val,A,N,T,CFG):
    y_pred = rf_clf.predict(x_val,A,N,T,False)
    mse = mean_squared_error(y_val,y_pred)
    return -mse

def optimize(space, rf_clf, x_val,y_val,CFG,fold,fun, n_calls=50):
    @use_named_args(space)
    def score(**params):
        final_score = fun(rf_clf,x_val,y_val,CFG=CFG, **params)
        return -final_score
    return gp_minimize(func=score, dimensions=space, n_calls=n_calls)


def run_config(CFG):
    start = time.time()
    skf = CFG.kfold(n_splits=CFG.n_folds, random_state=CFG.random_state,shuffle=True)
    df = CFG.df.fillna(0).reset_index(drop=True)
    if CFG.preprocces:
        df = CFG.preprocces(df)
    N_df,fet = df.shape
    train_features = df.drop(CFG.label_col, axis=1)
    labels = df[CFG.label_col]
    history = []
    y_preds= []
    y_true= []
    y_preds_without_random = []
    skf_split = skf.split(train_features,labels) if CFG.kfold == StratifiedKFold else skf.split(train_features)
    for fold,(train_index, val_index) in enumerate(tqdm(skf_split)):
        if fold not in CFG.trn_folds: continue
        x_train,y_train = train_features.loc[train_index],labels.loc[train_index]
        x_val,y_val = train_features.loc[val_index],labels.loc[val_index]
        
        ds_train = Dataset(df.loc[train_index], label=CFG.label_col, cat_features=[])
        ds_test =  Dataset(df.loc[val_index],  label=CFG.label_col, cat_features=[])
        
#         rf_clf = CFG.model(random_state=CFG.random_state)
        rf_clf = CFG.model(random_state=CFG.random_state,min_samples_split=N_df//100,min_samples_leaf=N_df//100,max_leaf_nodes=300)
        rf_clf.fit(x_train,y_train)
        
        space = [Real(0.0, 0.5, name='A'),Integer(1, 100, name='N'),Real(0.0, 0.2, name='T')] # Bayesian optimization for sensitivity analysis
        if CFG.model == DecisionTreeRegressor or CFG.model == DecisionTreeRegressor_KatzArbili:
            # print(f"fold {fold} start Bayesian optimization Reg")
            opt_result = optimize(space,rf_clf, x_val,y_val,CFG,fold,fun=calculate_final_score_reg, n_calls=CFG.n_calls)
            # dump(opt_result,f"opt/{CFG.d_name}_fold_{fold}.pkl")
            A = opt_result.x[0]
            N = opt_result.x[1]
            T = opt_result.x[2]
            y_pred = rf_clf.predict(x_val,A,N,T,to_print=False)
            y_pred_without_random = rf_clf.predict(x_val,0,1,0,to_print=False)
            auc = mean_squared_error(y_val,y_pred)
            acc = r2_score(y_val,y_pred)
            
            auc2 = mean_squared_error(y_val,y_pred_without_random)
            acc2 = r2_score(y_val,y_pred_without_random)
        else:
            # print(f"fold {fold} start Bayesian optimization class")
            opt_result = optimize(space,rf_clf, x_val,y_val,CFG,fold,fun=calculate_final_score_class, n_calls=CFG.n_calls)
            # dump(opt_result,f"opt/{CFG.d_name}_fold_{fold}.pkl")
            A = opt_result.x[0]
            N = opt_result.x[1]
            T = opt_result.x[2]
            y_pred = rf_clf.predict_proba(x_val,A,N,T,to_print=False)
            y_pred_without_random = rf_clf.predict_proba(x_val,0,1,0,to_print=False)
            auc = roc_auc_score(y_val,y_pred if CFG.multi_class else y_pred[:,1],multi_class='ovo' if CFG.multi_class else "raise")
            acc = accuracy_score(y_val.astype(int),y_pred.argmax(1).astype(int))
            
            auc2 = roc_auc_score(y_val,y_pred_without_random if CFG.multi_class else y_pred_without_random[:,1],multi_class='ovo' if CFG.multi_class else "raise")
            acc2 = accuracy_score(y_val.astype(int),y_pred_without_random.argmax(1).astype(int))
        history.append({"model":rf_clf,'y_pred':y_pred,"y_pred_without_random":y_pred_without_random,
                        "auc":auc,"acc":acc,"y_val":y_val,"auc_base":auc2,"acc_base2":acc2,
                        "alpha":A,"iter":N,"opt_result":opt_result,"ds_train":ds_train,"ds_test":ds_test})
        y_preds.append(y_pred)
        y_true.append(y_val)
        y_preds_without_random.append(y_pred_without_random)
    run_time = time.time()-start
    if CFG.model == DecisionTreeRegressor or CFG.model == DecisionTreeRegressor_KatzArbili:
        cv_auc = mean_squared_error(np.concatenate(y_true),np.concatenate(y_preds))
        cv_acc = r2_score(np.concatenate(y_true),np.concatenate(y_preds))
        
        cv_auc_without_random = mean_squared_error(np.concatenate(y_true),np.concatenate(y_preds_without_random))
        cv_acc_without_random = r2_score(np.concatenate(y_true),np.concatenate(y_preds_without_random))
        msg = f"datasets {CFG.d_name}, val oof mean_squared_error score {cv_auc},val oof r2 score {cv_acc},\
        mean_squared_error alpha 0 - score {cv_auc_without_random}, val oof r2 score alpha 0 {cv_acc_without_random} \
        run_time {run_time}, total size {N_df}, num fet {fet}"
        print(msg)
        history.append(msg)
    else:
        cv_auc = roc_auc_score(np.concatenate(y_true),np.concatenate(y_preds) if CFG.multi_class else np.concatenate(y_preds)[:,1],multi_class='ovo' if CFG.multi_class else "raise")
        cv_acc = accuracy_score(np.concatenate(y_true).astype(int),np.concatenate(y_preds).argmax(1).astype(int))
        
        cv_auc_without_random = roc_auc_score(np.concatenate(y_true),np.concatenate(y_preds_without_random) if CFG.multi_class else np.concatenate(y_preds_without_random)[:,1],multi_class='ovo' if CFG.multi_class else "raise")
        msg = f"datasets {CFG.d_name}, val oof auc score {cv_auc}, auc score without random  - {cv_auc_without_random}, run_time {run_time}, total size {N_df}, num fet {fet}"
        print(msg)
        history.append(msg)
    return history,cv_auc,cv_acc,run_time,df