import pandas as pd
import numpy as np

# Read in the excel file
# Returns:
#   X: first column is 1s, the rest are from the spreadsheet
#   Y: The last column from the spreadsheet
#   labels: The list of headers for the columns of X from the spreadsheet
def read_excel_data(infilename):
    ## Your code here
    df_infilename = pd.read_excel(infilename, index_col="property_id")
    Y = df_infilename["price"].to_numpy()
    X = df_infilename.drop("price", axis=1)
    labels = X.columns.to_list()
    ones = np.ones([len(X), 1])
    X = np.append(ones, X.to_numpy(), axis=1)
    B = np.linalg.inv(X.T @ X) @ X.T @ Y

    return X, Y, labels


# Make it pretty
def format_prediction(B, labels):
    ## Your code here
    pred_string = f"${round(B[0],2):,} + (${round(B[1],2)} X {labels[0]}) + (${round(B[2],2)} X {labels[1]}) + (${round(B[3],2):,} X {labels[2]}) + (${round(B[4],2):,} X {labels[3]}) + (${round(B[5],2):,} X {labels[4]})"
    return pred_string


# Return the R2 score for coefficients B
# Given inputs X and outputs Y
def score(B, X, Y):
    ## Your code here
    # y_pred = (X*B.Transpose()).sum(axis=1)
    y_pred = np.dot(X, B.T)
    y_mean = Y.mean()
    numerator = ((Y - y_pred) ** 2).sum()
    denominator = ((Y - y_mean) ** 2).sum()
    R2 = 1 - (numerator / denominator)
    return R2
