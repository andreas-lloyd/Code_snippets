from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Fix our state
seed = 100

# Some default parameters
rf_default = {
    'n_estimators' : [120, 300, 500, 800, 1200],
    'max_depth' : [5, 8, 15, 25, 30, None],
    'min_samples_split' : [1.0, 2, 5, 10, 15, 100],
    'min_samples_leaf' : [1, 2, 5, 10],
    'max_features' : ['log2', 'sqrt', None]
}

xgb_default = {
    'learning_rate' : [0.01, 0.015, 0.025, 0.05, 0.1],
    'gamma' : [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1], # can also try between the first two
    'max_depth' : [3, 5, 7, 9, 12, 15, 17, 25],
    'min_child_weight' : [1, 3, 5, 7],
    'subsample' : [0.6, 0.7, 0.8, 0.9, 1],
    'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1],
    'reg_lambda' : [0.01, 0.1, 1], # Also some random ones
    'reg_alpha' : [0, 0.1, 0.5, 1] # Also some random ones
}

logit_default = {
    'penalty' : ['l1', 'l2'],
    'C' : [0.001, 0.01, 0.1, 1, 10, 100]
}

def optimise_model(model, params, X, y, scoring = 'accuracy', refit = True, cv = 3, save_name = None):
    '''
    Wrapper for using gridsearch cv to optimise a model - the first argument is any model object and then params is the parameter dictionary
    Just doing this because find it a bit easier to always have in one place
    NOTE that the default for returning is the BEST model - set refit to False if don't want this
    '''
    optimise_grid = GridSearchCV(model, params, scoring = scoring, cv = cv, refit = refit, n_jobs = -1, verbose = 0)
    
    print('Starting the grid search...')
    optimise_grid.fit(X, y)
    
    print('Finished the grid search - the best score was {} for a parameter combination of \n{}'.format(optimise_grid.best_score_, optimise_grid.best_params_))
    
    # Will save to pickle if desired
    if save_name:
        with open(save_name, 'wb') as pickle_file:
            pickle.dump(optimise_grid, pickle_file)
    
    return optimise_grid

# Examples of how to use for the three defaults above - note that X and y are NOT provided

## RF
rf_model = RandomForestClassifier(random_state = seed)
rf_optimum = optimise_model(rf_model, rf_default, X, y)

## XGB
xgb_model = XGBClassifier(random_state = seed)
xgb_optimum = optimise_model(xgb_model, xgb_default, X, y)

## LR
lr_model = LogisticRegression(random_state = seed)
lr_optimum = optimise_model(lr_model, logit_default, X, y)