import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from data_formatting import data_form
from models import get_classification_model
from export import zindi_submission, prepare_for_submission 
from transform_targets import get_transf_targets

def run():
	# Import the Datasets
	train_data = pd.read_csv("/content/drive/My Drive/Regression Challenge Shared Folder/Data/Train.csv", index_col="ID") 
	test_data = pd.read_csv("/content/drive/My Drive/Regression Challenge Shared Folder/Data/Test.csv", index_col="ID")

	# Data formatting
	X_train_n, y_train_n, X_test, y_test, cat_cols, num_cols = data_form(train_data, test_data)    
    
    # Split on data. UNCOMMENT if use of get_transf_targets
    #X_train, X_valid, y_train, y_valid = train_test_split(X_train_n, y_train_n, train_size=0.8, test_size=0.2)
    
    #Comment if split is done
	X_train = X_train_n
    y_train = y_train_n
    
    # Build the model including preprocessing
	model = get_classification_model(cat_cols, num_cols)
	model.fit(X_train,y_train)

    #UNCOMMENT if use of y_tweak
    '''
    y_tweak = get_transf_targets(y_valid)
    y_pred = model.predict(X_valid)
    #log_loss on changed targets
    print(log_loss(y_tweak, y_pred))
    #log_loss on unchanged targets
    print(log_loss(y_valid, y_pred))
    '''

	# Predict
	y_pred = model.predict(X_test)
	y_pred = prepare_for_submission(y_pred, y_test)
	output_path = "/content/drive/My Drive/Regression Challenge Shared Folder/Outputs/log_reg_results.csv"

	# Export to csv file
	zindi_submission(y_pred, output_path)


if __name__ == "__main__":
	run()

