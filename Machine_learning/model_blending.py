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
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin


class blender(BaseEstimator, ClassifierMixin):
    '''
    Class that is used to blend various models together and predict on some test set
    '''

    def __init__(self, model_list = [{'log_reg' : LogisticRegression()}], n_folds = 3, scorer = accuracy_score, get_argmax = True, save_loc = None, seed = 100):
        '''
        What we initially provide when we declare it
        param: model_list will be a list of structure [{"Model_1" : model_1, "Model_2" : model_2, ...], {"Model_1" : model_1, "Model_2" : model_2, ...} ...] where each new list is a new level - and each level is a dictionary
        param: n_folds is the number of folds that it will use to generate the predictions
        param: scorer is the type of scorer we want to use
        param: get_argmax is for whether or not we should get the np.argmax for the scorer - will error if wrong (could fix this and make more robust)
        param: save_loc will be, if provided, the location to save CSV and pickls 
        '''
        assert type(model_list[0]) == dict, 'Have not detected a list of dicts from model_list - must wrap it up!'
        
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
        NOTE that can provide a dictionary to train_data, if want to use different training sets for different models - key names must match up with model!
        NOTE that train_y is assumed to already be encoded - could fix this later and add in here
        '''
        # Do a check on the length of train data
        assert type(train_data) != dict or len(train_data) == len(self.model_list[0]), 'Number of training sets not equal to number of models in first layer'
        assert type(train_data) != dict or train_data.keys() == self.model_list[0].keys(), 'The key names differ for the training data and model list provided'
        
        # Need to set attributes for each level and model to be able to use grid search well - will do this using the loop and FIXING names ourselves
        # Note that this MUST be done here to work properly when defaulting
        for level_num, level in enumerate(self.model_list):
            level_key = 'Level_{}_'.format(level_num + 1) # Do this because the levels have a real sense of order

            # Then we loop over the keys - this is crucial because the way we train models has no real sense of order
            for model in level:
                model_key = level_key + model

                # Then we set an attribute of the form self.Level_X_log_reg - which will allow us to use GridSearchCV
                setattr(self, model_key, level[model])
                
        self.train_data = train_data
        self.train_is_dic = type(train_data) == dict
        self.train_y = train_y
        self.num_classes = len(pd.unique(train_y))

        # Get the kfolds to be used throughout
        k_folds = StratifiedKFold(self.n_folds, random_state = self.seed)

        # Need to declare an initial training prediction dictionary - will be used for training throughout - will update with pd frames
        self.train_prediction_dic = {}
        
        # Then we start going through the levels - looping over the model list
        for level_num, level in enumerate(self.model_list):

            # First pick out the data we will use - unless we have a list
            if level_num == 0 and not self.train_is_dic:
                X = self.train_data # Use the raw training data for first level
            else:
                X = self.train_prediction_dic['Level_{}'.format(level_num)] # Otherwise use the predictions - note that start with level 1, not level 0...

            # Secondly will declare an empty pandas frame that we will use to store results
            level_results = pd.DataFrame()

            # Then start training each model
            for model_num, model in enumerate(level):
                print('Entering model number {} for level number {}'.format(model_num + 1, level_num + 1))
                # If the level is 0 and we have a list then we have to get the right data set (do it like this to avoid repeated assignment)
                if level_num == 0 and self.train_is_dic:
                    X = self.train_data[model]

                # Declare a pandas frame where we can put the results for each fold - which we will later concat to rest of results
                model_results = pd.DataFrame({'Model_{}_class_{}'.format(model_num + 1, class_num + 1) : [0]*len(self.train_y) for class_num in range(self.num_classes)})

                # Initialise a score list
                fold_scores = []

                # Then we use our splits to generate the folds
                for train_index, val_index in k_folds.split(X, self.train_y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = self.train_y.iloc[train_index], self.train_y.iloc[val_index]

                    # Retrieve the model correctly from attributes
                    model_key = 'Level_{}_{}'.format(level_num + 1, model)
                    getattr(self, model_key).fit(X_train, y_train)

                    # Last step is to predict on our "validation" set, and then save these results in our frame
                    fold_predictions = getattr(self, model_key).predict_proba(X_val)
                    
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
                getattr(self, model_key).fit(X, self.train_y)

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
        assert (type(test_data) == dict and self.train_is_dic) or (type(test_data) != dict and not self.train_is_dic)

        # Then again check the length of our test data
        assert type(test_data) != dict or len(test_data) == len(self.model_list[0]), 'Number of test sets not equal to number of models in first layer'

        # And again check the key names
        assert type(test_data) != dict or test_data.keys() == model_list[0].keys() 

        self.test_data = test_data

        # Declare a prediction dictionary to store our results
        self.test_prediction_dic = {}

        # Then we start going through the levels in our trained model list
        for level_num, level in enumerate(self.model_list):

            # Again pick out our data as long as it is not a list
            if level_num == 0 and not self.train_is_dic:
                X = self.test_data # Use the raw training data for first level
            else:
                X = self.test_prediction_dic['Level_{}'.format(level_num)] # Otherwise use the predictions

            # Secondly will declare an empty pandas frame that we will use to store results
            level_results = pd.DataFrame()

            # Then start fitting using each model
            for model_num, model in enumerate(level):
                print('Entering model number {} for level number {}'.format(model_num + 1, level_num + 1))
                
                # Similarly - if we trained on a list then we need to get the first set of test data
                if level_num == 0 and self.train_is_dic:
                        X = self.test_data[model]

                # Like before we will declare a model results frame to save the predictions
                model_key = 'Level_{}_{}'.format(level_num + 1, model)
                model_probabilities = getattr(self, model_key).predict_proba(X) # Might need to review this step
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



def get_optimise_grid(model_list, param_grid, copies_dict = {}):
    '''
    Helper function that lets us use cross validation to optimise the structure and parameters of our blender
    Have to feed in the basic structure to model list, as normal, then also a parameter dictionary which takes the form
    Level_X__<tag>__<argument>
    Then optionally a structure dictionary which lets us vary the structure, that is, how many times we should include a model in a given layer
    Each layer combination will be tested for all the possible parameter arguments
    Similar structure to param_grid, but does not take the __<argument> - and the list it points to is the number of variations of ways to include the given model
    The return value can be used with the optimise_function wrapper for use with grid search
    '''
    # Get the number of levels
    num_levels = len(model_list)

    # Initialise what we are going to feed into our GridSearch at the end
    optimise_params = {'model_list' : []}

    # Set up the keys we will be using
    model_keys = ['Level_{}__{}'.format(level_num + 1, model) for level_num, level in enumerate(model_list) for model in level]

    # Fill in the gaps in copies_dict
    for model_key in [model_key for model_key in model_keys if model_key not in copies_dict]:
        copies_dict[model_key] = [1]

    # And will fill in the gaps in the params dictionary - just so don't have to modify what comes later
    param_models = ['__'.join(key.split('__', 1)) for key in param_grid]
    for level_num, level in enumerate(model_list):
        for model in level:
            if 'Level_{}__{}'.format(level_num + 1, model) not in param_models:
                # Get the defaults
                default_params = model_list[level_num][model].get_params()
                first_default = list(default_params.keys())[0]
                
                # Then add in
                param_grid['Level_{}__{}__{}'.format(level_num + 1, model, first_default)] = [default_params[first_default]]

    # Now create expanded copies_dict to get the different structures - then can create new param grids for each "structure"
    for structure in ParameterGrid(copies_dict):
        
        #print(structure)
        
        # Have to check that there is no layer that has 0 and then has a layer above with non zero
        num_models = [sum([structure[model] for model in structure if 'Level_{}'.format(level_num + 1) in model]) for level_num in range(num_levels)]
        zero_check = sum([num_models[level_num] == 0 and num_models[level_num + 1] == 0 for level_num in range(num_levels - 1)])
        
        # Do not enter if there are no models in the first layer
        if zero_check == 0:
            # Start our with an empty dictionary that we will add to
            temp_param_grid = {}

            # Go for all non zero valued models and start expanding grid
            for model in structure:
                # Pull out the parameters from param_grid (in part) then split on __ - and add in a tag for which copy it is - and then we update our temp grid
                to_add = {'__copy_{}__'.format(copy + 1).join(key.split('__', 1)) : param_grid[key] for key in param_grid if model in key for copy in range(structure[model])}
                temp_param_grid.update(to_add)

            # Now that we have our temporary grid set up with ONLY the models to include - we can declare the host of models for this structure - each combination is its own model
            # pull the level and pull the tag
            for params in ParameterGrid(temp_param_grid):
                
                #print(params)
                
                # Get the different things we will be modifying
                level_model_set = set(['__'.join(key.split('__')[0:3]) for key in params])
                
                # Will generate a "model_dic" for each combination - dictionary for now just for the sake of the order
                temp_model_dic = {}
                
                # Loop over the different models we will be looking at
                for level_model in level_model_set:
                    # Get identifiers                
                    level_num = level_model.split('__')[0]
                    copy =  level_model.split('__')[1]
                    model_code = level_model.split('__')[2]

                    # Get a dictionary in the form <argument> : value for the current level and model combination
                    arguments = {argument.split('__')[3] : params[argument] for argument in params if level_model in argument}
                    
                    # Then get the model out - point to the level in the list (level_num - 1) - must clone or updates all!
                    model = clone(model_list[int(level_num.split('_')[1]) - 1][model_code])
                    model.set_params(**arguments)
                    
                    # Then update our temporary model dic on the specific level
                    temp_code = model_code + '__' + copy
                    if level_num in temp_model_dic:
                        temp_model_dic[level_num].update({temp_code : model})
                    else:
                        temp_model_dic[level_num] = {temp_code : model}
                
                # Now to convert to the correct format - extract numbers and then order
                level_nums = sorted([int(level_num.split('_')[1]) for level_num in temp_model_dic])
                temp_model_list = [temp_model_dic['Level_{}'.format(level_num)] for level_num in level_nums]
                #print(temp_model_list)
                #print('\n')
                
                # And finally add to our final output thing
                if len(optimise_params['model_list']) == 0:
                    optimise_params['model_list'] = [temp_model_list] # I have to do this to make sure it stays as a list of lists
                else:
                    optimise_params['model_list'].append(temp_model_list)

    return optimise_params



### EXAMPLES ###
model_list = [
    {
        'rf' : RandomForestClassifier(random_state = 100),
        'xgb' : XGBClassifier(random_state = 100),
        'lr' : LogisticRegression(random_state = 100)
    },
    {
        'xgb' : XGBClassifier(random_state = 200)
    }
]

param_grid = {
    'Level_1__rf__n_estimators' : [120, 300, 500],
    'Level_1__xgb__n_estimators' : [120, 300, 500],
    'Level_2__xgb__n_estimators' : [120, 300, 500]
}

copies_dict = {
    'Level_1__rf' : [0, 1, 2],
    'Level_1__xgb' : [0, 1, 2],
    'Level_2__xgb' : [0, 1]    
}

def custom_predict(P):
    return P.median(axis = 1)

predictions = blender_model.predict(X_v, custom_predict)
accuracy_score(y_v, predictions)