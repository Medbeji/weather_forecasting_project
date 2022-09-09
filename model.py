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



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# save the model to disk
import pickle 


def preprocess_dataset(data: pd.DataFrame):
    """
        applying the preprocessing of datasets following the following steps: 
        - normalizing data
        - removing outliers
        - inputing nulls categorical values
    """

    data_df = data.copy()

    # normalizing rain today and tommorow columns 
    data_df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
    data_df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

    # Imputer les variables catégorielle avec la méthode mode()
    subsets = []

    for is_raining in [0, 1]:
        temp_df = data_df[data_df['RainTomorrow'] == is_raining]
        temp_df['WindGustDir'] = temp_df['WindGustDir'].fillna(temp_df['WindGustDir'].mode()[0])
        temp_df['WindDir9am'] = temp_df['WindDir9am'].fillna(temp_df['WindDir9am'].mode()[0])
        temp_df['WindDir3pm'] = temp_df['WindDir3pm'].fillna(temp_df['WindDir3pm'].mode()[0])

        # Removing outliers 
        Q1 = temp_df.quantile(0.25)
        Q3 = temp_df.quantile(0.75)
        IQR = Q3 - Q1
        temp_df = temp_df[~((temp_df < (Q1 - 1.5 * IQR)) | (temp_df > (Q3 + 1.5 * IQR))).any(axis=1)]

        # append the values to subsets  
        subsets.append(temp_df)

    return pd.concat(subsets)


def apply_oversampling(data: pd.DataFrame): 
    """ Handling class imbalance using oversampling method """
    # handling class imbalance
    no = data[data.RainTomorrow == 0]
    yes = data[data.RainTomorrow == 1]
    yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
    oversampled = pd.concat([no, yes_oversampled])
    return oversampled
    

data = pd.read_csv("data/weatherAus.csv", header=0, squeeze=True)

FEATURES = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']


target = 'RainTomorrow'

X_train, X_test, y_train, y_test = train_test_split(data, data[target], test_size=0.3, random_state=13)

X_train = preprocess_dataset(data=X_train)
X_test = preprocess_dataset(data=X_test)

X_train = apply_oversampling(data= X_train)

y_train = X_train[target]
X_train = X_train[FEATURES]
y_test = X_test[target]
X_test = X_test[FEATURES]


numeric_features =  X_train.select_dtypes(include= np.number).columns
numeric_transformer = Pipeline(steps=[('imputer', IterativeImputer()),
    ('scaler', MinMaxScaler())])


categorical_features = X_train.select_dtypes(include=['object']).columns
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


print('Training the models.. with X shape = ', X_train.shape, 'y shape', y_train.shape)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred) 

print(accuracy)
print(roc_auc)
print(classification_report(y_test,y_pred,digits=5))


filename = 'model_19_08.sav'
pickle.dump(clf, open('models/' + filename, 'wb'))



""""
------------------------------- CASE 1 : 70/30 -------------------------------------

    Dataset split Train 70% / Test 30%
    Scores généré sont : 
    accuracy => 0.8973683394811072
    roc_auc => 0.8651836705828564

   ------------------- classification report -------------------------- 
                precision    recall  f1-score   support

         0.0    0.93971   0.92291   0.93123     24996
         1.0    0.75675   0.80201   0.77872      7475

    accuracy                        0.89508     32471
   macro avg    0.84823   0.86246   0.85498     32471
weighted avg    0.89759   0.89508   0.89613     32471

------------------------------------------------------------------------------ 
------------------------------- CASE 1 : 80/20 -------------------------------------

    Dataset split Train 80% / Test 20%
    Scores généré sont : 
    accuracy => 0.8997535686055703
    roc_auc => 0.8662161076595627

   ------------------- classification report -------------------------- 
    
               precision    recall  f1-score   support

         0.0    0.93865   0.92330   0.93091     16637
         1.0    0.75640   0.79783   0.77656      4966

    accuracy                        0.89446     21603
   macro avg    0.84752   0.86056   0.85374     21603
weighted avg    0.89675   0.89446   0.89543     21603


"""
