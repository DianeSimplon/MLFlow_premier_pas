 # create a DistanceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


from utils import haversine_vectorized

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
                
        # A COMPPLETER
        self.start_lat= start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
    
    def fit(self, X, y=None):
        # A COMPLETER retourne l'objet lui meme 
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        #Calculer la distance
        X_[['distance']]= haversine_vectorized(X_)
        return X_[['distance']]


# create a TimeFeaturesEncoder
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.

    """

    def __init__(self, pickup_datetime="pickup_datetime"):

       # A COMPLETER
        self.pickup_datetime=pickup_datetime


    def fit(self, X, y=None):
        # A COMPLETER
        return self

    def transform(self, X, y=None):
        # A COMPLETER 
        X_ = X.copy()
        X_["pickup_datetime"] = pd.to_datetime(X_["pickup_datetime"]).dt.tz_convert("America/New_York")
        X_["dow"] = X_["pickup_datetime"].dt.dayofweek
        X_["hour"] = X_["pickup_datetime"].dt.hour
        X_["month"] = X_["pickup_datetime"].dt.month
        X_["year"] = X_["pickup_datetime"].dt.year
        return X_[['dow', 'hour', 'month', 'year']]


