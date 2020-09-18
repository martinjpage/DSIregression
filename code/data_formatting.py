import pandas as pd
from clustering import get_customer_cluster

def data_form(train_data, test_data):
    '''This function takes in the training dataset and test dataset and delivers the
    training and validation dataframes as well a list of the catergorical and numerical columns'''

    #Define target columns
    y_columns = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ','GYSR', 'SOP4', 'RVSZ',
                 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X','K6QO', 'QBOL',
                 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']

    # Remove rows with missing target
    train_data.dropna(axis=0, subset=y_columns, inplace=True)
    test_data.dropna(axis=0, subset=y_columns, inplace=True)

    # convert date using pandas
    train_data['join_date'] = pd.to_datetime(train_data['join_date'])
    test_data['join_date'] = pd.to_datetime(test_data['join_date'])

    # add age column (== birth_year)
    train_data['age'] = 2020 - train_data["birth_year"]
    test_data['age'] = 2020 - test_data["birth_year"]

    # add age_joined column (age of client when joined)
    train_data['age_join'] = train_data['join_date'].dt.year - train_data["birth_year"]
    test_data['age_join'] = test_data['join_date'].dt.year - test_data["birth_year"]

    # period_client (== join_date in years; duration)
    train_data['period_client'] = 2020 - train_data['join_date'].dt.year
    test_data['period_client'] = 2020 - test_data['join_date'].dt.year

    # Convert string date to seconds
    train_data['join_date'] = (train_data['join_date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    test_data['join_date'] = (test_data['join_date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # FEATURE number of products - Count the products choosen by a user
    train_data["number_product"] = train_data[y_columns].sum(axis=1)
    test_data["number_product"] = test_data[y_columns].sum(axis=1) + 1

    # customer_cluster (group customer is classified into using GMM)
    train_data['customer_cluster'] = get_customer_cluster(train_data)
    test_data['customer_cluster'] = get_customer_cluster(test_data)

    # format categorical columns
    train_data["sex"] = train_data["sex"].astype('category')
    train_data["marital_status"] = train_data["marital_status"].astype('category')
    train_data["branch_code"] = train_data["branch_code"].astype('category')
    train_data["occupation_category_code"] = train_data["occupation_category_code"].astype('category')
    train_data["occupation_code"] = train_data["occupation_code"].astype('category')
    train_data["number_product"] = train_data["number_product"].astype('category')
    train_data['customer_cluster'] = train_data['customer_cluster'].astype('category')

    test_data["sex"] = test_data["sex"].astype('category')
    test_data["marital_status"] = test_data["marital_status"].astype('category')
    test_data["branch_code"] = test_data["branch_code"].astype('category')
    test_data["occupation_category_code"] = test_data["occupation_category_code"].astype('category')
    test_data["occupation_code"] = test_data["occupation_code"].astype('category')
    test_data["number_product"] = test_data["number_product"].astype('category')
    test_data['customer_cluster'] = test_data['customer_cluster'].astype('category')

    #Separate target from predictors
    y_train = train_data[y_columns]
    X_train = train_data.drop(y_columns, axis = 1)

    X_test = test_data.drop(y_columns, axis = 1)
    y_test = test_data[y_columns]


    # Select categorical columns and numerical columns
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype.name == "category"]
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]


    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train[my_cols].copy()
    X_test = X_test[my_cols].copy()

    return X_train, y_train, X_test, y_test, categorical_cols, numerical_cols
