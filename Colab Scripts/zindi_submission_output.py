def zindi_submission(predictions, filename):
    '''
    Function takes as input a dataframe of predictions with an index that represents
    IDs, column names of different insurance products, and values of probablities,
    as well as a filename (string) for the output file
    Function returns a csv file in a two column format: one with the ID index and
    product name concatenated as a string and a corresponding second column with the
    probablity values as a float, which is called "label
    '''

    # promote index as a column
    predictions = predictions.reset_index()
    # extract name of the index/id column
    index_name = predictions.columns[0]
    # reshape dataframe with long format (melting) - "product" for the column names;
    # "label" for probability values
    melted_preds = predictions.melt(id_vars=index_name, var_name="Product", value_name="Label")

    # concatenate the ID and product name columns into a series
    id_x_pcode = melted_preds[index_name] + " X " + melted_preds["Product"]
    # extract the label column for the melted dataframe
    label_series = melted_preds[["Label"]]
    # bind the ID-Product name column with the label column into a dataframe; give column names
    final = pd.concat([id_x_pcode, label_series], axis=1, ignore_index=True)
    final.columns = ["ID X PCODE", "Label"]
    # export to CSV
    final.to_csv(filename, index=False, header=True)