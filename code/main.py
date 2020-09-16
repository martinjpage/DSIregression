import pandas as pd
from data_formatting import data_form
from models import get_classification_model
from export import zindi_submission, prepare_for_submission 


def run():
	# Import the Datasets
	train_data = pd.read_csv("/content/drive/My Drive/Regression Challenge Shared Folder/Data/Train.csv", index_col="ID") 
	test_data = pd.read_csv("/content/drive/My Drive/Regression Challenge Shared Folder/Data/Test.csv", index_col="ID")

	# Data formatting
	X_train, y_train, X_test, y_test, cat_cols, num_cols = data_form(train_data, test_data)

	# Build the model including preprocessing
	model = get_classification_model(cat_cols, num_cols)
	model.fit(X_train,y_train)

	# Predict
	y_pred = model.predict(X_test)
	y_pred = prepare_for_submission(y_pred, y_test)
	output_path = "/content/drive/My Drive/Regression Challenge Shared Folder/Outputs/log_reg_results.csv"

	# Export to csv file
	zindi_submission(y_pred, output_path)


if __name__ == "__main__":
	run()

