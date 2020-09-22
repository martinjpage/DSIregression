from catboost import CatBoostClassifier
from export import zindi_submission


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
zindi_submission(y_pred, ""DSIregression/Outputs/catboost.csv")
