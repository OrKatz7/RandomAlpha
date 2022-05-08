import dabl
import pandas as pd
from models_3 import DecisionTreeRegressor_KatzArbili,DecisionTreeClassifier_KatzArbili
from sklearn.model_selection import StratifiedKFold,KFold


DEBUG = False
class music:#class
    d_name = 'music'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv('input/data/music-genre-classification/train.csv')
    drop_col = ['Artist Name','Track Name']
    df = df.drop(drop_col,1)
    df['time_signature'] = df['time_signature'].astype('float')
    label_col = 'Class'
    if DEBUG: 
        df = df.sample(n=500,random_state=0).reset_index(drop=True)
    model = DecisionTreeClassifier_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
#     preprocces = dabl.clean
    preprocces = False
    multi_class = True
    kfold = StratifiedKFold
    n_calls = 100
    
class ionosphere:#class
    d_name = 'ionosphere'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv('input/data/ionosphere/ionosphere_data.csv')
    df.loc[df["column_ai"] == "g", "column_ai"] = 0
    df.loc[df["column_ai"] == "b", "column_ai"] = 1
    label_col = 'column_ai'
    if DEBUG: 
        df = df.sample(n=500,random_state=0).reset_index(drop=True)
    model = DecisionTreeClassifier_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
    preprocces = None
    multi_class = False
    kfold = StratifiedKFold
    n_calls = 10
    
class santander_customer_satisfaction: #class
    d_name = 'santander_customer_satisfaction'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv("input/data/santander-customer-satisfaction/train.csv")
    drop_col = ['ID']
    df = df.drop(drop_col,1)
    label_col = "TARGET"
    model = DecisionTreeClassifier_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
    preprocces = None
    multi_class = False
    kfold = StratifiedKFold
    n_calls = 100
    
class Wids2021: #class
    d_name = 'Wids'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv("input/data/training-data/TrainingWiDS2021.csv")
    drop_col = ['hospital_id','ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type','Unnamed: 0','encounter_id']
    df = df.drop(drop_col,1)
    label_col = "diabetes_mellitus"
    if DEBUG: 
        df = df.sample(n=50000,random_state=0).reset_index(drop=True)
    model = DecisionTreeClassifier_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
    preprocces = dabl.clean
    multi_class = False
    kfold = StratifiedKFold
    n_calls = 100
    
class fetal_health: #class
    d_name = 'fetal_health'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv("input/data/fetal-health-classification/fetal_health.csv")
#     df.loc[df["gender"] == "Male", "gender"] = 0
#     df.loc[df["gender"] == "Female", "gender"] = 1
    label_col = "fetal_health"
    df[label_col] = df[label_col]-1
    model = DecisionTreeClassifier_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
    preprocces = None
    multi_class = True
    kfold = StratifiedKFold
    n_calls = 100
    
class santander: #reg
    d_name = 'santander'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv('input/data/santander-value-prediction-challenge/train.csv')
    drop_col = ['ID']
    df = df.drop(drop_col,1)
    label_col = "target"
    df[label_col] = df[label_col]/1000000
    model = DecisionTreeRegressor_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
    preprocces = dabl.clean
    multi_class = True
    kfold = KFold
    n_calls = 100
    
class California: #reg
    d_name = 'California'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv('input/data/california-housing-prices-data-extra-features/California_Houses.csv')
    label_col = "Median_House_Value"
    df[label_col] = df[label_col]/1000000
    model = DecisionTreeRegressor_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
    preprocces = dabl.clean
    multi_class = True
    kfold = KFold
    n_calls = 100
    
class playground: #reg
    d_name = 'playground'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv("input/data/tabular-playground-series-jan-2021/train.csv")
    df = df[['cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10','cont11','cont12','cont13','cont14','target']]
    df = df[:1000].copy()
    label_col = "target"
    model = DecisionTreeRegressor_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
    preprocces = dabl.clean
    multi_class = True
    kfold = KFold
    n_calls = 100
    
class Wine: #reg
    d_name = 'Wine'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv("input/data/wine-quality-dataset/WineQT.csv")
    label_col = "quality"
    model = DecisionTreeRegressor_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
    preprocces = dabl.clean
    multi_class = True
    kfold = KFold
    n_calls = 100
    
class Medicalpremium: #reg
    d_name = 'Medical premium'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv("input/data/medical-insurance-premium-prediction/Medicalpremium.csv")
    label_col = "PremiumPrice"
    df["PremiumPrice"] = df["PremiumPrice"]  / 10000.0
    model = DecisionTreeRegressor_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
    preprocces = dabl.clean
    preprocces = False
    multi_class = True
    kfold = KFold
    n_calls = 100
    
class Avocado: #reg
    d_name = 'Avocado'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv("input/data/avacado-price-prediction/Avocado.csv")
    drop_col = ['Unnamed: 0', 'Date', 'type', 'region','year']
    df = df.drop(drop_col,1)
#     if DEBUG: 
#         df = df.sample(n=500,random_state=0).reset_index(drop=True)
    label_col = "AveragePrice"
    model = DecisionTreeRegressor_KatzArbili
    history = None
    cv_auc = None
    cv_acc = None
    run_time = None
#     preprocces = dabl.clean
    preprocces = False
    multi_class = True
    kfold = KFold
    n_calls = 100