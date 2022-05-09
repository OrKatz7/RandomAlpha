# Random Alpha for Decision Tree
This repository represents open-source research developed by [Ofir Arbili](https://www.linkedin.com/in/ofir-arbili-82375179/?originalSubdomain=il/), [Or Katz](https://www.linkedin.com/in/or-katz-9ba885114/)

# TL;DR
In this project, a random function is added to the Decision Tree split function during inference in order to improve decision tree accuracy.
The work is divided into three parts:
In the first part, the classification decision tree split function changed such that at each split, there will be an alpha probability (e.g., alpha=10%) of being routed in the opposite direction to what the condition indicates, and a 1-alpha probability (i.e., 1-alpha=90%) of being routed according to the condition. The modified prediction algorithm is also run n times for each sample, and the probability vectors are averaged to provide a final prediction. The second part make the same modification but to Decision tree regressor.
The third part adds additional condition to the first and second part. The random split condition should be used only when the feature value is at most x percent higher or lower than the split value.  The main concept of this addition is to use the random function only when the feature is close to the node condition threshold to generate more accurate “soft split”.
Five regression datasets and five classification datasets were used to test the modified Decision Tree algorithms. In our study, we analyzed diversified datasets related to healthcare, economics, and signal processing. Our results in parts 1 and 2 indicated modest improvement. Part 3 of the study showed constant improvements but would vanish if other more complicated tree algorithms were used instead of the Decision Tree algorithm.

## Input and Output data
download the data from https://drive.google.com/drive/folders/1IlAjfCeoYZP0Nh9m3Dcuna8dIOo0awGO?usp=sharing

## Train + Inference
1. run Q1_2/main.ipynb
2. run Q3/main.ipynb
3. run eval.ipynb
4. deepchecks_.ipynb


## Config - Q1_2/configs.py, Q3/configs.py

config format example:
```` 
class Wids2021:
    d_name = 'Wids'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    random_state = 42
    verbose=False
    deepchecks=True
    df = pd.read_csv("../input/data/training-data/TrainingWiDS2021.csv")
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
```` 

## Classification Results:

| Dataset | AUC - Baseline | AUC - rnd_split (Q1) | AUC - rnd_splt_w_th(Q3) |
| --------------- | --------------- | --------------- | --------------- |
| music | 0.892 | 0.897 | **0.899** |
| ionosphere | 0.919| 0.899 |  **0.935** |
| santander_customer_sat | 0.821 | 0.82 | **0.824** |
| Wids | 0.812 | 0.815 | **0.821** |
| fetal_health | 0.908 | **0.937** |  **0.937** |

## Regression Results:

| Dataset |  MSE - Baseline |  MSE - rnd_split (Q1) |  MSE - rnd_splt_w_th(Q3) |
| --------------- | --------------- | --------------- | --------------- |
| Santander | 58.041 | 57.487 | **57.185** |
| California | 0.004 | 0.004 |  **0.399** |
| Wine | 0.479 | 0.425 | **0.412** |
| Medical premium | 0.085 | 0.087 | **0.081** |
| Avocado | 0.076 | 0.076 |  **0.074** |

## Sensitivity Analysis
Our sensitivity analysis was performed using Bayesian optimization when we defined the search sword as 2 parameters in part 1 and 2 (alpha and N) and 3 parameters in part 3 (alpha, N, and threshold).This method produces better results than fixing variables and attempting to optimize them separately. We used SKOPT to minimize MSE in the regression task and to minimize 1-AUC in the classification task.

example:
![alt text](https://github.com/OrKatz7/RandomAlpha/blob/main/docs/Wids_SA.png)

## Conclusion


