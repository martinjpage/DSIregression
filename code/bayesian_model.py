from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pymc3 as pm

def get_bayesian_model(cat_cols, num_cols):

    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Preprocessing for categorical data

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)])

    with pm.Model() as linear_model:
        weights = pm.Normal('weights', mu=0, sigma=1)
        noise = pm.Gamma('noise', alpha=2, beta=1)
        y_observed = pm.Normal('y_observed',
                           mu=0,
                           sigma=10,
                           observed=y_test)

        prior = pm.sample_prior_predictive()
        posterior = pm.sample()
        posterior_pred_clf = pm.sample_posterior_predictive(posterior)

        # Bundle preprocessing and modeling code in a pipeline
        model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', posterior_pred_clf)])

    return model