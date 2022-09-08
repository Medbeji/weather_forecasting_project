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

        temp_df['Date'] = temp_df['Date'].fillna(temp_df['Date'].mode()[0])
        temp_df['Location'] = temp_df['Location'].fillna(temp_df['Location'].mode()[0])
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

data = preprocess_dataset(data=data)

target = 'RainTomorrow'

X_train, X_test, y_train, y_test = train_test_split(data, data[target], test_size=0.2, random_state=13)

X_train = apply_oversampling(data= X_train)

y_train = X_train[target]
X_train = X_train[FEATURES]
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


filename = 'model_09_08.sav'
pickle.dump(clf, open('models/' + filename, 'wb'))



""""

    Dataset split Train 70% / Test 30%
    Scores généré sont : 
    accuracy => 0.8973683394811072
    roc_auc => 0.8651836705828564
    classification report => 
                                precision    recall  f1-score   support

                            0.0    0.94214   0.92405   0.93300    24950
                            1.0    0.75674   0.80632   0.78074     7311

                    accuracy                           0.89737    32261
                    macro avg      0.84944   0.86518   0.85687    32261
                    weighted avg   0.90012   0.89737   0.89850    32261
    ---------------------------------------------------------------------------

    Dataset split Train 80% / Test 20%
    Scores généré sont : 
    accuracy => 0.8997535686055703
    roc_auc => 0.8662161076595627
    classification report => 
                                          precision    recall  f1-score   support

                                    0.0    0.94180   0.92764   0.93467     16625
                                    1.0    0.76559   0.80479   0.78470      4882

                                accuracy                       0.89975     21507
                            macro avg      0.85369   0.86622   0.85968     21507
                            weighted avg   0.90180   0.89975   0.90063     21507


"""
