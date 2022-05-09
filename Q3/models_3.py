import pandas as pd
import numpy as np
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
from joblib import Parallel, delayed


class DecisionTreeClassifier_KatzArbili(DecisionTreeClassifier):
    
    '''
    A recursive predict fucntion

        Parameters:
            x (array): The samples data for prediction
            node(int): Current node in the tree
            depth(int): Current depth of the tree
            node_feature_id (list): A list of all nodes and the feature used for the split on this node.
            in_alpha (float): The probability of taking the opposite direction of what the node condition indicates.
            in_rnd_list(list): A list for each node indicate if to use the correct of the opposite direction of the node.
            to_print(bool): Use for debugging puroposes - print each step
        Returns:
            predict (np.array): The calcualted prediction for each sample
    '''
    def recurse_predict(
        self,
        x,
        node,
        depth,
        node_feature_id,
        in_alpha,
        in_rnd_list,
        in_min_ratio_threshold,
        to_print,
    ):
        try:
            if self.tree_.feature[node] != _tree.TREE_UNDEFINED:
                feature_id = node_feature_id[node]
                feature_threshold = self.tree_.threshold[node]
                if to_print:
                    print(
                        "feature_id {}. feature_threshold:{}, feature_value:{},rnd:{}".format(
                            feature_id,
                            feature_threshold,
                            x[feature_id],
                            in_rnd_list[node],
                        )
                    )
                if bool((x[feature_id] <= feature_threshold)) != bool(
                    (in_rnd_list[node] == 1)
                    and (
                        abs(x[feature_id] - feature_threshold) / feature_threshold
                        < in_min_ratio_threshold
                    )
                ):
                    if to_print:
                        print("took left node")
                    ret_val = self.recurse_predict(
                        x,
                        self.tree_.children_left[node],
                        depth + 1,
                        node_feature_id,
                        in_alpha,
                        in_rnd_list,
                        in_min_ratio_threshold,
                        to_print,
                    )
                else:
                    if to_print:
                        print("took right node")
                    ret_val = self.recurse_predict(
                        x,
                        self.tree_.children_right[node],
                        depth + 1,
                        node_feature_id,
                        in_alpha,
                        in_rnd_list,
                        in_min_ratio_threshold,
                        to_print,
                    )
                if ret_val is not None:
                    return ret_val
            else:
                return self.tree_.value[node][0] / self.tree_.value[node][0].sum()
        except Exception as e:
            print("Error when calling {} function".format("recurse_predict"))
            raise e
            
    
    '''
    A warpper function for the recursive predict fucntion

        Parameters:
            x (array): The samples data for prediction
            in_alpha (float): The probability of taking the opposite direction of what the node condition indicates.
            in_node_feature_id (list): A list of all nodes and the feature used for the split on this node.
            in_node_use_random_arr(list): A list for each node indicate if to use the correct of the opposite direction of the node.
            to_print(bool): Use for debugging puroposes - print each step
        Returns:
            predict (np.array): The calcualted prediction for each sample
    '''
    def predict_proba_one(
        self,
        x,
        in_alpha,
        in_node_feature_id,
        in_node_use_random_arr,
        in_min_ratio_threshold,
        to_print,
    ):
        try:
            return self.recurse_predict(
                x,
                0,
                1,
                in_node_feature_id,
                in_alpha,
                in_node_use_random_arr,
                in_min_ratio_threshold,
                to_print,
            )
        except Exception as e:
            print("Error when calling {} function".format("predict_proba_one"))
            raise e
    
    '''
    This function iterate for each sample and calculate the predicted probabilty. For performace issues, before calling the predict_proba_one function, this 
    function calculated for all nodes and all samples the indication of whether using the corerct direction or the opposite direction of the node.

        Parameters:
            x (array): The samples data for prediction
            in_alpha (float): The probability of taking the opposite direction of what the node condition indicates.
            to_print(bool): Use for debugging puroposes - print each step
        Returns:
            predict (np.array): The calcualted prediction for each sample
    '''
    def predict_proba_rnd(self, x, in_alpha, in_min_ratio_threshold, to_print):
        try:
            # list of all nodes and which feature is used for the split on the specific node
            node_feature_id = [
                i if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in self.tree_.feature
            ]
            
            # array which holds for all nodes and all samples the indication of whether using the corerct direction or the opposite direction of the node.
            node_use_random_arr = np.random.choice(
                a=[False, True],
                size=(len(x.index), len(node_feature_id)),
                p=[1 - in_alpha, in_alpha],
            )

            # convert all samples data to numeric
            x = x.apply(pd.to_numeric, errors="coerce")
            
            # itereate over the samples and call predict for each sample
            ret_val = []
            for ind, x_row in enumerate(x.values):
                if to_print:
                    print("x_row", x_row)
                ret_val.append(
                    self.predict_proba_one(
                        x_row,
                        in_alpha,
                        node_feature_id,
                        node_use_random_arr[ind],
                        in_min_ratio_threshold,
                        to_print,
                    )
                )
                
            return np.array(ret_val)
        except Exception as e:
            print("Error when calling {} function".format("predict_proba_rnd"))
            raise e
    
    '''
    This is an override on the legend function predict_proba, this function call the predict_proba_rnd (which get alpha as input and randomly with the
    probability of alpha decided to use the opposite branch of the node).

        Parameters:
            x (array): The samples data for prediction
            in_alpha (float): The probability of taking the opposite direction of what the node condition indicates.
            in_n (int):  in_n times (e.g. n=100) for each sample and then average the probability vectors to provide a final prediction.
            to_print(bool): Use for debugging puroposes - print each step
        Returns:
            predict (np.array): The calcualted prediction for each sample
    '''
    def predict_proba(self, x, in_alpha, in_n, in_min_ratio_threshold=0.15, to_print=False):
        try:
            ret_val = Parallel(n_jobs=48)(delayed(self.predict_proba_rnd)(x,in_alpha,in_min_ratio_threshold,to_print) for ind in range(in_n))
            return np.average(np.array(ret_val), axis=0)
        except Exception as e:
            print("Error when calling {} function".format("predict_proba"))
            raise e

class DecisionTreeRegressor_KatzArbili(DecisionTreeRegressor):

    # section 1-2
    '''
    A recursive predict fucntion

        Parameters:
            x (array): The samples data for prediction
            node(int): Current node in the tree
            depth(int): Current depth of the tree
            node_feature_id (list): A list of all nodes and the feature used for the split on this node.
            in_alpha (float): The probability of taking the opposite direction of what the node condition indicates.
            in_rnd_list(list): A list for each node indicate if to use the correct of the opposite direction of the node.
            to_print(bool): Use for debugging puroposes - print each step
        Returns:
            predict (np.array): The calcualted prediction for each sample
    '''
    def recurse_predict(
        self, x, node, depth, node_feature_id, in_alpha, in_rnd_list, in_min_ratio_threshold, to_print
    ):
        try:
            if self.tree_.feature[node] != _tree.TREE_UNDEFINED:
                feature_id = node_feature_id[node]
                feature_threshold = self.tree_.threshold[node]
                if to_print:
                    print(
                        "feature_id {}. feature_threshold:{}, feature_value:{}, rnd:{}".format(
                            feature_id,
                            feature_threshold,
                            x[feature_id],
                            in_rnd_list[node] == 1,
                        )
                    )

                if bool((x[feature_id] <= feature_threshold)) != bool(
                        (in_rnd_list[node] == 1)
                        and (
                            abs(x[feature_id] - feature_threshold) / feature_threshold
                            < in_min_ratio_threshold)
                        ):
                    if to_print:
                        print("took left node")
                    ret_val = self.recurse_predict(
                        x,
                        self.tree_.children_left[node],
                        depth + 1,
                        node_feature_id,
                        in_alpha,
                        in_rnd_list,
                        in_min_ratio_threshold,
                        to_print,
                    )
                else:
                    if to_print:
                        print("took right node")
                    ret_val = self.recurse_predict(
                        x,
                        self.tree_.children_right[node],
                        depth + 1,
                        node_feature_id,
                        in_alpha,
                        in_rnd_list,
                        in_min_ratio_threshold,
                        to_print,
                    )
                if ret_val is not None:
                    return ret_val
            else:
                return self.tree_.value[node][0]
        except Exception as e:
            print("Error when calling {} function".format("recurse_predict"))
            raise e    
    
    '''
    A warpper function for the recursive predict fucntion

        Parameters:
            x (array): The samples data for prediction
            in_alpha (float): The probability of taking the opposite direction of what the node condition indicates.
            in_node_feature_id (list): A list of all nodes and the feature used for the split on this node.
            in_node_use_random_arr(list): A list for each node indicate if to use the correct of the opposite direction of the node.
            in_min_ratio_threshold: the max precentage diffrence from feature value to feature threshold value to use the opposite direction
            to_print(bool): Use for debugging puroposes - print each step
        Returns:
            predict (np.array): The calcualted prediction for each sample
    '''
    def predict_proba_one(
        self, x, in_alpha, in_node_feature_id, in_node_use_random_arr, in_min_ratio_threshold, to_print
    ):
        
        try:
            if to_print:
                print("x {}".format(x))
            # calling tree in recursive manner
            return self.recurse_predict(
                x, 0, 1, in_node_feature_id, in_alpha, in_node_use_random_arr, in_min_ratio_threshold, to_print
            )
        except Exception as e:
            print("Error when calling {} function".format("predict_proba_one"))
            raise e
    
    '''
    This function iterate for each sample and calculate the predicted probabilty. For performace issues, before calling the predict_proba_one function, this 
    function calculated for all nodes and all samples the indication of whether using the corerct direction or the opposite direction of the node.

        Parameters:
            x (array): The samples data for prediction
            in_alpha (float): The probability of taking the opposite direction of what the node condition indicates.
            in_min_ratio_threshold: the max precentage diffrence from feature value to feature threshold value to use the opposite direction
            to_print(bool): Use for debugging puroposes - print each step
        Returns:
            predict (np.array): The calcualted prediction for each sample
    '''
    def predict_rnd(self, x, in_alpha, in_min_ratio_threshold, to_print):
        try:
            
            # list of all nodes and which feature is used for the split on the specific node
            node_feature_id = [
                i if i != _tree.TREE_UNDEFINED else "undefined!" for i in self.tree_.feature
            ]
            
            # array which holds for all nodes and all samples the indication of whether using the corerct direction or the opposite direction of the node.
            node_use_random_arr = np.random.choice(
                a=[False, True],
                size=(len(x.index), len(node_feature_id)),
                p=[1 - in_alpha, in_alpha],
            )
            
            # convert all samples data to numeric
            x = x.apply(pd.to_numeric, errors="coerce")
            
            # itereate over the samples and call predict for each sample
            ret_val = []
            for ind, x_row in enumerate(x.values):
                ret_val.append(
                    self.predict_proba_one(
                        x_row, in_alpha, node_feature_id, node_use_random_arr[ind], in_min_ratio_threshold, to_print
                    )[0]
                )
                
            return np.array(ret_val)
        except Exception as e:
            print("Error when calling {} function".format("predict_rnd"))
            raise e
    
    '''
    This is an override on the legend function predict, this function call the predict_rnd (which get alpha as input and randomly with the
    probability of alpha decided to use the opposite branch of the node).

        Parameters:
            x (array): The samples data for prediction
            in_alpha (float): The probability of taking the opposite direction of what the node condition indicates.
            in_n (int):  in_n times (e.g. n=100) for each sample and then average the probability vectors to provide a final prediction.
            in_min_ratio_threshold: the max precentage diffrence from feature value to feature threshold value to use the opposite direction
            to_print(bool): Use for debugging puroposes - print each step
        Returns:
            predict (np.array): The calcualted prediction for each sample
    '''
    def predict(self, x, in_alpha, in_n, in_min_ratio_threshold=0.15, to_print=False):
        try:
            ret_val = Parallel(n_jobs=48)(delayed(self.predict_rnd)(x,in_alpha,in_min_ratio_threshold,to_print) for ind in range(in_n))
            return np.average(np.array(ret_val), axis=0)
        except Exception as e:
            print("Error when calling {} function".format("predict"))
            raise e