from catboost import CatBoostClassifier
from export import zindi_submission
from data_formatting import simple_data_form

train_path = "DSIregression/Data/Train.csv"
test_path = "DSIregression/Data/Test.csv"
output_path = "DSIregression/Outputs/catboost.csv"

train_data = pd.read_csv(train_path, index_col="ID")
test_data = pd.read_csv(test_path, index_col="ID")

X_train, y_train, X_test = simple_data_form(train_data, test_data)

model = CatBoostClassifier(task_type="GPU")
model.fit(X_train, y_train, cat_features=categorical_col)

# Predictions
y_pred = model.predict_proba(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = target_encoder.inverse_transform(y_pred.columns)
y_pred = y_pred.set_index(X_test.index)

X_test_product = X_test[y_pred.columns]
y_pred[X_test_product == 1] = 1 # set 1 where it was already 1
y_pred

# Export to csv for  submission
zindi_submission(y_pred, output_path)
