import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR, SVR
#Relier les pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def get_data(nrows=10000):
    '''returns a DataFrame with nrows'''
    df = pd.read_csv('../data/train.csv',  nrows=nrows)
    # A COMPLETER
    return df

# implement clean_data() function
def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    # A COMPLETER
    df = df[df.fare_amount > 0]
    #df = df[df.distance < 100]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count != 0]
    
    return df