from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


class ProbLogisticRegression(LogisticRegression):
    """
    Model wrapper to solve the cross-validation problem
    predict method is overloaded to output probabilities of getting the label 1
    """
    def predict(self, X):
        return LogisticRegression.predict_proba(self, X)[:, 1] # we only need the prob to get the label 1
# Preprocessing for numerical data

def get_classification_model(cols):
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
            ('num', numeric_transformer, cols['numerical_cols']),
            ('cat', categorical_transformer, cols['categorical_cols'])]) 
    
    # Optimized for hyperparameters, no cross validation, Multi-target output classifier
    multi_target_clf = MultiOutputClassifier(ProbLogisticRegression(penalty='l2', C=1,  max_iter=400), n_jobs=-1)
    
    # Bundle preprocessing and modeling code in a pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', multi_target_clf)])
    return model