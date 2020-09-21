#import libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer

#import functions
from data_formatting import data_form
from export import zindi_submission

#set file paths
train_path = "DSIregression/Data/Train.csv"
test_path = "DSIregression/Data/Test.csv"
output_path = "DSIregression/Outputs/results/csv"

def get_data():
    # Import the Datasets
    train_data = pd.read_csv(train_path, index_col="ID")
    test_data = pd.read_csv(test_path, index_col="ID")
	# Data formatting
	X_train, y_train, X_test, y_test, cat_cols, num_cols = data_form(train_data, test_data)
        return X_train, y_train, X_test, y_test, cat_cols, num_cols

X_train, y_train, X_test, y_test, cat_cols, num_cols = get_data()

# Preprocessing for numerical data

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)])

## Random Forest Model
SEED = 2

class ProbRandomForestClassifier(RandomForestClassifier):
    """
    Model wrapper to solve the cross-validation problem
    predict method is overwritten to output probabilities of getting the label 1
    """
    def predict(self, X):
        train_prediction = RandomForestClassifier.predict_proba(self, X)
        train_prediction = np.array(train_prediction)[:, :, 1]
        train_prediction = train_prediction.transpose()
        return train_prediction


rf = ProbRandomForestClassifier(random_state=SEED)

#setup pipeline to bundle preprocessing and modeling code
rf_steps = [('preprocessor', preprocessor), ('model', rf)]
rf_pipeline = Pipeline(rf_steps)

#define grid parameters
# params_rf = {'model__max_depth': [3, 8, 15, None],
#               'model__max_features': ['sqrt', 'log2'],
#               "model__min_samples_leaf": [1, 2, 5, 10],
#               "model__min_samples_split": [2, 5, 10, 15],
#               'model__n_estimators': [100, 300, 500, 800]}

params_rf = {'model__max_depth': [3, 8, 15, None],
              'model__max_features': ['sqrt', 'log2', 'auto'],
              "model__min_samples_leaf": [1, 5, 10],
              "model__min_samples_split": [5, 15, 30, 40],
              'model__n_estimators': [100, 300]}

#instantiate grid
grid_rf = GridSearchCV(estimator = rf_pipeline, param_grid = params_rf, cv = 3, verbose = 2, n_jobs = None, scoring=make_scorer(log_loss, needs_proba=False)) #change to -1 for parallel
grid_rf.fit(X_train, y_train)

### Training Prediction
grid_rf.best_params_
# {'model__max_depth': 3, 'model__max_features': 'log2', 'model__min_samples_leaf': 10, 'model__min_samples_split': 30, 'model__n_estimators': 100}

print("Log loss:", grid_rf.best_score_)
#3.6410452567769354

### Training PredictionTest prediction
y_pred = best_rf.predict_proba(X_test)
y_pred = np.array(y_pred)[:, :, 1]
y_pred = y_pred.transpose()
y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)

y_pred[y_test == 1] = 1
y_pred

#CSV output
zindi_submission(y_pred, output_path)

##Zindi score: 0.0936251630934617 (ID: 7yRnFT4A)