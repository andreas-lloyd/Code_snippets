import pandas as pd

def treat_nulls(pandas_frame):
    '''
    Function to treat the nulls from a data set - also adds variables that are used as part of that
    '''
    print('Have found {} nulls'.format(sum(pandas_frame.isnull().any())))

    ## CODE ##

    print('Now have {} nulls'.format(sum(pandas_frame.isnull().any())))
    return pandas_frame

def create_variables(pandas_frame):
    '''
    Function to add variables to our data set
    '''
    ## CODE ##
    print('After creating variables, now have shape {}'.format(pandas_frame.shape))
    return pandas_frame

def transform_data(train, target, test = None, nulls = False, variables = False, drop_columns = None, get_dummies = False):
    '''
    Just a basic wrapper that will use every now and again to encapsulate the data transformation
    If test is provided, then the data is concatenated
    Can control which processes are done using the logical arguments - but note that these functions must be
    defined by the user in each case
    Note that the null removal is done first - so if want to add in any null type variables, have to add them in
    '''
    print('Starting out with a train set of shape {}'.format(train.shape))

    # Remove the target and leave us with only the features
    y_train = train[target]
    transformed_data = train.drop(target, axis = 1)

    if test:
        print('Have found a test set of shape {} - combining now'.format(test.shape))
        transformed_data = pd.concat([transformed_data, test])
        print('The combined data has shape {}'.format(transformed_data.shape))

    if nulls:
        print('Dealing with nulls...')
        transformed_data = treat_nulls(transformed_data)

    if variables:
        print('Making new variables...')
        transformed_data = create_variables(transformed_data)

    if drop_columns:
        print('Dropping columns...')
        transformed_data.drop(drop_columns, axis = 1, inplace = True)
        print('After dropping columns have shape {}'.format(transformed_data.shape))

    if get_dummies:
        print('Getting dummies...')
        transformed_data = pd.get_dummies(transformed_data, drop_first = True)
        print('After getting dummies have shape {}'.format(transformed_data.shape))

    if test:
        return y_train, transformed_data[0:train.shape[0]], transformed_data[train.shape[0]:]
    else:
        return y_train, transformed_data