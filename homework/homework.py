# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#

# Carga de librerias
import pandas as pd 
from sklearn.model_selection import GridSearchCV 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle
import numpy as np
import os
import json
import time
import gzip

def clean_data(data_dfp):
    dfp=data_dfp.copy()
    dfp=dfp.rename(columns={'default payment next month': 'default'})
    dfp=dfp.drop(columns='ID')
    dfp['EDUCATION'] = dfp['EDUCATION'].replace(0, np.nan)
    dfp['MARRIAGE'] = dfp['MARRIAGE'].replace(0, np.nan)
    dfp=dfp.dropna()
    dfp.loc[dfp['EDUCATION'] > 4, 'EDUCATION'] = 4
    return dfp

def get_features_target(data, target_column):
    x = data.drop(columns=target_column)
    y = data[target_column]
    return x, y




def create_pipeline(dfp):
 
    fcat = ['SEX', 'EDUCATION', 'MARRIAGE']
    fnums = [col for col in dfp.columns if col not in fcat]

    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), fcat),
            ('num', StandardScaler(), fnums)
        ]
    )

   
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('pca', PCA()),
            ('select_k_best', SelectKBest(f_classif)),
            ('model', SVC())
        ]
    )

    return pipeline





def optimize_hyperparameters(pipeline, x_train, y_train):

    gp = {
        'pca__n_components': [21],
        'select_k_best__k': [12],
        'model__C': [0.8],
        'model__kernel': ['rbf'],
        'model__gamma': [0.1],
      

    }
    sg = GridSearchCV(pipeline, gp, cv=10, scoring='balanced_accuracy', n_jobs=-1, verbose=2)
    sg.fit(x_train, y_train)

    
    return sg


#
def save_model(model):
    
    if not os.path.exists('files/models'):
        os.makedirs('files/models')
    
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': float(round(precision_score(y_train, y_train_pred),3)),
        'balanced_accuracy': float(round(balanced_accuracy_score(y_train, y_train_pred),3)),
        'recall': float(round(recall_score(y_train, y_train_pred),3)),
        'f1_score': float(round(f1_score(y_train, y_train_pred),3))
    }

    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': float(round(precision_score(y_test, y_test_pred),3)),
        'balanced_accuracy': float(round(balanced_accuracy_score(y_test, y_test_pred),3)),
        'recall': float(round(recall_score(y_test, y_test_pred),3)),
        'f1_score': float(round(f1_score(y_test, y_test_pred),3))
    }

    print(metrics_train)
    print(metrics_test)

    return metrics_train, metrics_test



def calculate_confusion_matrix(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    cm_matrix_train = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        'true_1': {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])}
    }

    cm_matrix_test = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        'true_1': {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])}
    }

    return cm_matrix_train, cm_matrix_test

if __name__ == '__main__':
    
 
    train_data_zip = 'files/input/train_data.csv.zip'
    test_data_zip = 'files/input/test_data.csv.zip'

    
    train_data=pd.read_csv(
        train_data_zip,
        index_col=False,
        compression='zip')

    test_data=pd.read_csv(
        test_data_zip,
        index_col=False,
        compression='zip')
    
    
    train_data=clean_data(train_data)
    test_data=clean_data(test_data)

    
    x_train, y_train = get_features_target(train_data, 'default')
    x_test, y_test = get_features_target(test_data, 'default')

    

    
    pipeline = create_pipeline(x_train)

    
    start = time.time()
    model = optimize_hyperparameters(pipeline, x_train, y_train)
    end = time.time()
    print(f'Time to optimize hyperparameters: {end - start:.2f} seconds')

    print(model.best_params_)

   
    save_model(model)

    
    metrics_train, metrics_test = calculate_metrics(model, x_train, y_train, x_test, y_test)

    
    cm_matrix_train, cm_matrix_test = calculate_confusion_matrix(model, x_train, y_train, x_test, y_test)

    print(cm_matrix_train)

    
    if not os.path.exists('files/output'):
        os.makedirs('files/output')

    
    metrics = [metrics_train, metrics_test, cm_matrix_train, cm_matrix_test]
    pd.DataFrame(metrics).to_json('files/output/metrics.json', orient='records', lines=True)