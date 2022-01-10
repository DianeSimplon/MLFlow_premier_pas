 # create a DistanceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import data
from data import get_data , clean_data
import encoders
from encoders import DistanceTransformer
from encoders import TimeFeaturesEncoder
import utils
from utils import compute_rmse

class Trainer:

    def __init__(self,):
        # A COMPLETER
        self.data = clean_data(get_data(nrows=10000))
       


    # implement set_pipeline() function
    def set_pipeline():
        '''returns a pipelined model'''
        # A  COMPLETER
    
        pipe_distance = Pipeline([
            ('DistanceTransformer', DistanceTransformer()),
            ('StandarScaler', StandardScaler()),
        ])

        # create time pipeline time_pipe
        time_pipe =Pipeline([
            ('datetime', TimeFeaturesEncoder()),
            ('encoder', OneHotEncoder()),

        ])
        preproc_pipe = ColumnTransformer(transformers=[('distance', pipe_distance,
                                    ['pickup_latitude', 'pickup_longitude',
                                    'dropoff_latitude', 'dropoff_longitude']),
                                    
                                    ('time', time_pipe,
                                    ['pickup_datetime'])])
        pipe = Pipeline([
        ('preproc_pipe', preproc_pipe),
        ('regression', LinearRegression())
        ])
        return pipe


    # implement train() function
    def run(X_train, y_train, pipeline_model):
        '''returns a trained pipelined model'''
        pipeline_model.fit(X_train, y_train)
    
    # A COMPLETER
        return pipeline_model



    # implement evaluate() function
    def evaluate(X_test, y_test, pipeline):
        '''returns the value of the RMSE'''
        # A COMPLETER
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


objet = Trainer()
y = objet.data["fare_amount"]
X = objet.data.drop("fare_amount", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=32)

pipeline = objet.set_pipeline()
objet.run(X_train, y_train, pipeline)
rmse = objet.evaluate(X_test, y_test, pipeline)
print(rmse) 