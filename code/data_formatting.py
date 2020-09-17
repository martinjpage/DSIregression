import time


def data_form(train_data, test_data):
    '''This function takes in the training dataset and test dataset and delivers a '''

    #Define target columns
    y_columns = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ','GYSR', 'SOP4', 'RVSZ',
                 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X','K6QO', 'QBOL',
                 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']

    # Convert string date to seconds
    train_data['join_date'] = train_data['join_date'].apply(lambda d: time.mktime (time.strptime(str(d), "%d/%m/%Y")) if str(d) !='nan' else float('nan'))
    test_data['join_date'] = test_data['join_date'].apply(lambda d: time.mktime (time.strptime(str(d), "%d/%m/%Y")) if str(d) !='nan' else float('nan'))

    # FEATURE number of products - Count the products choosen by a user
    train_data["number_product"] = train_data[y_columns].sum(axis=1)
    test_data["number_product"] = test_data[y_columns].sum(axis=1) + 1

    #Separate target from predictors
    y_train = train_data[y_columns]
    X_train = train_data.drop(y_columns, axis = 1)

    X_test = test_data.drop(y_columns, axis = 1)
    y_test = test_data[y_columns]


    # Select categorical columns and numerical columns
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]


    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train[my_cols].copy()
    X_test = X_test[my_cols].copy()

    return X_train, y_train, X_test, y_test, categorical_cols, numerical_cols
