'''
Will make a type of class that can be used to blend several models together - based on something I saw on Kaggle
The idea will be that the class is initialised with several models for each layer, which are then trained on folds of the data
The result will be class probabilities for the test set which can be maniuplated however we want

Explanaition:
    Training:
        When we train we initial train the level 0 with the actual training data - later levels are trained with the predictions of the previous level
        Then what we will do is SPLIT the data into a sub-training and validation set, which we will use to generate predictions and models
        At each step we will save the model that was trained
        Note that the predictions made must match up with the indices of the validation set
        We must train on the same splits etc. for every model (that is we have a for model... for fold... type loop)

        NOTE that we could also add functionality that allows us to have DIFFERENT training sets for different models - could be useful

        Finally should save all models and save final predictions to a CSV

    Test:
        The first thing to do is to "re-train" our model using all of the data = note that could put this into the fit method easily: 
            - We first train the 0 level model on all of our data
            - Then we train the n > 0 level models on the predictions that we generated earlier
            - Then we use the models trained at each stage on the test data

        This is done to take full advantage at each step of all the data - instead of using a model trained on only parts of the data
        Although I guess there could be a bit of difference in that our probabilities are generated from "partial" models the first round,
        but not for the test set

NOTE that for now - if you want to use blender as a model input to blender itself - must set scorer to None (issue with pandas frames vs numpy arrays, could fix but...)

Have to think about:
    How we can optimise it (pass to CV)
    How we can pass parameters to the models

    How about giving the final classification method as an argument to the base classifier - instead of the predict method (but then also being able to give in predict method), this would allow for CV
'''
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin


class blender(BaseEstimator, ClassifierMixin):
    '''
    Class that is used to blend various models together and predict on some test set
    '''

    def __init__(self, model_list = [[LogisticRegression()]], n_folds = 3, scorer = accuracy_score, get_argmax = True, save_loc = None, seed = 100):
        '''
        What we initially provide when we declare it
        param: model_list will be a list of structure [[model_1, model_2, ...], [model_1, model_model_2 ...] ...] where each new list is a new level - Note that must be a list of lists
        param: n_folds is the number of folds that it will use to generate the predictions
        param: scorer is the type of scorer we want to use
        param: get_argmax is for whether or not we should get the np.argmax for the scorer - will error if wrong (could fix this and make more robust)
        param: save_loc will be, if provided, the location to save CSV and pickls 
        '''
        assert type(model_list[0]) == list, 'Have not detected a list of lists from model_list - must wrap it up!'
        self.model_list = model_list
        self.n_folds = n_folds
        self.scorer = scorer
        self.get_argmax = get_argmax
        self.save_loc = save_loc
        self.seed = 100



    def fit(self, train_data, train_y):
        '''
        The fit method - here we will generate our probabilities from the base data and then train models on top
        Train data and train y are the sets of data and targets for training
        NOTE that can provide a list to train_data, if want to use different training sets for different models
        NOTE that train_y is assumed to already be encoded - could fix this later and add in here
        '''
        # Do a check on the length of train data
        assert type(train_data) != list or len(train_data) == len(self.model_list[0]), 'Number of training sets not equal to number of models in first layer'

        self.train_data = train_data
        self.train_is_list = type(train_data) == list
        self.train_y = train_y
        self.num_classes = len(pd.unique(train_y))

        # Get the kfolds to be used throughout
        k_folds = StratifiedKFold(self.n_folds, random_state = self.seed)

        # Need to declare an initial training prediction dictionary - will be used for training throughout - will update with pd frames
        self.train_prediction_dic = {}
        
        # Then we start going through the levels - looping over the model list
        for level_num, level in enumerate(self.model_list):

            # First pick out the data we will use - unless we have a list
            if level_num == 0 and not self.train_is_list:
                X = self.train_data # Use the raw training data for first level
            else:
                X = self.train_prediction_dic['Level_{}'.format(level_num)] # Otherwise use the predictions - note that start with level 1, not level 0...

            # Secondly will declare an empty pandas frame that we will use to store results
            level_results = pd.DataFrame()

            # Then start training each model
            for model_num, model in enumerate(level):
                print('Entering model number {} for level number {}'.format(model_num + 1, level_num + 1))
                # If the level is 0 and we have a list then we have to get the right data set (do it like this to avoid repeated assignment)
                if level_num == 0 and self.train_is_list:
                    X = self.train_data[model_num]

                # Declare a pandas frame where we can put the results for each fold - which we will later concat to rest of results
                model_results = pd.DataFrame({'Model_{}_class_{}'.format(model_num + 1, class_num + 1) : [0]*len(self.train_y) for class_num in range(self.num_classes)})

                # Initialise a score list
                fold_scores = []

                # Then we use our splits to generate the folds
                for train_index, val_index in k_folds.split(X, self.train_y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = self.train_y.iloc[train_index], self.train_y.iloc[val_index]

                    model.fit(X_train, y_train)

                    # Last step is to predict on our "validation" set, and then save these results in our frame
                    fold_predictions = model.predict_proba(X_val)
                    
                    model_results.iloc[val_index,:] = fold_predictions
                    
                    # Now get our fold score
                    if self.get_argmax and self.scorer is not None:
                        fold_scores.append(self.scorer(y_val, np.argmax(fold_predictions, axis = 1)))
                    elif self.scorer is not None:
                        fold_scores.append(self.scorer(y_val, fold_predictions))
                    
                # Now we need to save the results to our data frame
                level_results = pd.concat([level_results, model_results], axis = 1)
                
                # Get the accuracy of our predictions for monitoring
                print('Model {} for level {} has an average score of {} with a std of {}'.format(model_num, level_num, np.mean(fold_scores), np.std(fold_scores)))

                # After having trained all the levels - need to train on the full data - we only fit on the data and nothing else (in the predict method we will .predict() for each model)
                model.fit(X, self.train_y)

            # Then finally we will save the level results to the full thing - ready for next round
            self.train_prediction_dic['Level_{}'.format(level_num + 1)] = level_results

            # And also will save this to CSV if desired (NOTE that will save a pickle later with the full dictionary of results)
            if self.save_loc:
                csv_name = str(Path(self.save_loc) / 'Level_{}_train_results.csv'.format(level_num + 1))
                level_results.to_csv(csv_name, index = False)


        # Will save the number of models in the last layer
        self.num_models_last_level = len(level)

        # Now save pickles of the list of models
        if self.save_loc:
            with open(str(Path(self.save_loc) / 'Model_list.pickle'), 'wb') as pickle_file:
                pickle.dump(self.model_list, pickle_file)


        return self

    def predict_proba(self, test_data):
        '''
        Here will predict on some test or validation data, getting the probabilities initially and then continuously predicting
        Using the models and levels we defined before
        '''
        # First make sure the test data looks like our test data
        assert (type(test_data) == list and self.train_is_list) or (type(test_data) != list and not self.train_is_list)

        # Then again check the length of our test data
        assert type(test_data) != list or len(test_data) == len(self.model_list[0]), 'Number of test sets not equal to number of models in first layer'

        self.test_data = test_data

        # Declare a prediction dictionary to store our results
        self.test_prediction_dic = {}

        # Then we start going through the levels in our trained model list
        for level_num, level in enumerate(self.model_list):

            # Again pick out our data as long as it is not a list
            if level_num == 0 and not self.train_is_list:
                X = self.test_data # Use the raw training data for first level
            else:
                X = self.test_prediction_dic['Level_{}'.format(level_num)] # Otherwise use the predictions

            # Secondly will declare an empty pandas frame that we will use to store results
            level_results = pd.DataFrame()

            # Then start fitting using each model
            for model_num, model in enumerate(level):
                print('Entering model number {} for level number {}'.format(model_num + 1, level_num + 1))
                
                # Similarly - if we trained on a list then we need to get the first set of test data
                if level_num == 0 and self.train_is_list:
                        X = self.test_data[model_num]

                # Like before we will declare a model results frame to save the predictions
                model_probabilities = model.predict_proba(X) # Might need to review this step
                model_results = pd.DataFrame(model_probabilities, columns = ['Model_{}_class_{}'.format(model_num + 1, class_num + 1) for class_num in range(self.num_classes)])

                # And save it to our level results
                level_results = pd.concat([level_results, model_results], axis = 1)

            # And finally save to our prediction dictionary for the next round
            self.test_prediction_dic['Level_{}'.format(level_num + 1)] = level_results

            # And also will save this to CSV if desired
            if self.save_loc:
                csv_name = str(Path(self.save_loc) / 'Level_{}_test_results.csv'.format(level_num + 1))
                level_results.to_csv(csv_name, index = False)

        # Will save the last level of predictions as that is where we will maniuplate
        self.last_level_predictions = self.test_prediction_dic['Level_{}'.format(level_num + 1)]
        
        # Then finally return the test prediction dictionary so the user can do what they want with it
        return self.last_level_predictions
    
    def predict(self, test_data, predict_function = 'max'):
        '''
        Predict method that will first call predict proba and then get some basic arg max or some other function on the data to return our final classification
        Can also take a prediction function that will combine the results as desired - must be of form f(P) where P is the probability matrix for a given class, 
        for example mean(P) would take mean of every column of class 0, 1, 2, ..., N - and we ALWAYS take an argmax of these at the end
        Can also give string arguments "mean" or "max" for the straight mean or argmax (defaults to argmax)
        '''

        assert predict_function in ['max', 'mean'] or callable(predict_function), 'The argument "predict_function" must be a callable or one of the default strings'

        # Save what we have used
        self.predict_function = predict_function

        # First get our probabilities
        print('Getting the prediction probabilities...')
        prediction_probabilities = self.predict_proba(test_data)

        transformed_probabilities = pd.DataFrame({'Class_{}'.format(class_num + 1) : [0]*prediction_probabilities.shape[0] for class_num in range(self.num_classes)})

        # Then we have to combine the probabilities in some way, taking into account how many classes we have
        print('Getting the final classification...')
        for class_num in range(self.num_classes):
            # First we get the list of columns we want to look at
            model_columns = ['Model_{}_class_{}'.format(model_num + 1, class_num + 1) for model_num in range(self.num_models_last_level)]
            
            # Then we will apply the desired function to these column
            if self.predict_function == 'max':
                transformed_probabilities['Class_{}'.format(class_num + 1)] = prediction_probabilities[model_columns].max(axis = 1)

            elif self.predict_function == 'mean':
                transformed_probabilities['Class_{}'.format(class_num + 1)] = prediction_probabilities[model_columns].mean(axis = 1)

            else:
                transformed_probabilities['Class_{}'.format(class_num + 1)] = self.predict_function(prediction_probabilities[model_columns])

        # To get the argmax have to do this stupid thing because of how idxmax works        
        self.prediction = transformed_probabilities.idxmax(axis = 1).replace('Class_', '', regex = True).astype(int) - 1

        return self.prediction

    def score(self, y_true, y_pred, scorer = None):
        if scorer is None:
            scorer = self.scorer
        
        return scorer(y_true, y_pred)



### EXAMPLES ###
def custom_predict(P):
    return P.median(axis = 1)

predictions = blender_model.predict(X_v, custom_predict)
accuracy_score(y_v, predictions)