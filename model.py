# Implémenter le meme process mais en utilisant les pipelines pour faciliter l'inférence
import colored_traceback
colored_traceback.add_hook()

import numpy as np 
import pandas as pd 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin #gives fit_transform method for free


# save the model to disk
import pickle 


def preprocess_dataset(data: pd.DataFrame, selected_features: list, target: str):
    """
        applying the preprocessing of datasets following the following steps: 
        - normalizing data
        - handling class imbalance
        - removing outliers
        - inputing nulls categorical values
    """

    # normalizing rain today and tommorow columns 
    data['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
    data['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

    # handling class imbalance
    no = data[data.RainTomorrow == 0]
    yes = data[data.RainTomorrow == 1]
    yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
    oversampled = pd.concat([no, yes_oversampled])

    # Imputer les variables catégorielle avec la méthode mode()
    oversampled['Date'] = oversampled['Date'].fillna(oversampled['Date'].mode()[0])
    oversampled['Location'] = oversampled['Location'].fillna(oversampled['Location'].mode()[0])
    oversampled['WindGustDir'] = oversampled['WindGustDir'].fillna(oversampled['WindGustDir'].mode()[0])
    oversampled['WindDir9am'] = oversampled['WindDir9am'].fillna(oversampled['WindDir9am'].mode()[0])
    oversampled['WindDir3pm'] = oversampled['WindDir3pm'].fillna(oversampled['WindDir3pm'].mode()[0])

    # Removing outliers 
    Q1 = oversampled.quantile(0.25)
    Q3 = oversampled.quantile(0.75)
    IQR = Q3 - Q1

    print(oversampled.shape)

    data_preprocessed = oversampled[~((oversampled < (Q1 - 1.5 * IQR)) |(oversampled > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    print(data_preprocessed.shape)
    
    # separating X & y
    X = data_preprocessed[FEATURES]
    y = data_preprocessed[target]

    print(' The shape of dataset after preprocessing ', X.shape)

    return X, y

data = pd.read_csv("data/weatherAus.csv", header=0, squeeze=True)
FEATURES = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']
X, y = preprocess_dataset(data=data, selected_features=FEATURES, target='RainTomorrow')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

numeric_features =  X.select_dtypes(include= np.number).columns
numeric_transformer = Pipeline(steps=[('imputer', IterativeImputer()),
    ('scaler', MinMaxScaler())])



class MyLabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.encoder = LabelEncoder()
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

    def fit_transform(self, x, y=0):
        return self.encoder.fit(x).transform(x)

categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
       ])


params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(**params_rf))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred) 

print(accuracy)
print(roc_auc)
print(classification_report(y_test,y_pred,digits=5))


filename = 'model_13_08.sav'
pickle.dump(clf, open('models/' + filename, 'wb'))