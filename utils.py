import pandas as pd

def fill_na_with_mean(df):
    ''' Replaces missing values with the mean.

    If the mean can't be computed, assume the column contains strings and
    leave as NaN.
    '''
    for column in df.columns.values:
        # Replace NaNs with the median
        #TODO: Replace ordinal with mode and don't replace cat
        try:
            df[column].fillna(df[column].median(), inplace=True)
        except TypeError:
            pass
    return df

def catagorical_features_to_columns(df):
    ''' Converts catagorical features to columns.

    Uses pandas get_dummies.
    One column for each categorical variable.
    Column name will be (original column name + '_' + variable name).
    So for example, the SaleCondition column would get converted into
    3 columns one column name would be 'SaleCondition_Partial' and this
    column would have a 1 for each sample that had 'Partial' in the
    original 'SaleCondition' column and a 0 for each sample that had
    something else in the 'SaleCondition' column.
    '''
    # Get a list of object columns (the categorical variables).
    col_types = df.columns.to_series().groupby(df.dtypes).groups
    col_types = {k.name: v for k, v in col_types.items()}
    obj_cols = list(col_types['object'])
    # Use get_dummies on each of these columns.
    for col in obj_cols:
        df = pd.get_dummies(df, columns=[col])
    return df

def remove_rare_features(df, cutoff=10):
    ''' Removes features where the minimum class is less than the cutoff.
    '''
    df.ix[:, df.columns] = df.ix[:, df.columns].apply(pd.to_numeric)
    df.drop([col for col, val in df.sum().iteritems() if val < cutoff], axis=1, inplace=True)
    
def match_columns(df1, df2):
    ''' Remove columns that don't apear in both dataframes.
    '''
    columns1 = df1.columns
    columns2 = df2.columns
    columns_both = list(set(columns1) & set(columns2))
    df1 = df1[columns_both]
    df2 = df2[columns_both]
    return [df1,df2]

def clean(df_train, df_test=None, custom_clean=None):
    ''' Cleans datasets.
    '''
    if custom_clean is not None:
        df_train = custom_clean(df_train)    
    df_train = fill_na_with_mean(df_train)    
    df_train = catagorical_features_to_columns(df_train)
    remove_rare_features(df_train)
    if df_test is None:
        return df_train
    else:
        if custom_clean is not None:
            df_test = custom_clean(df_test)
        df_test = fill_na_with_mean(df_test)
        df_test = catagorical_features_to_columns(df_test)
        df_train, df_test = match_columns(df_train, df_test)
        return [df_train, df_test]
    
'''
------------------------------------------------------------------------------------------
'''

# TODO: Split the clean and learn sections into multiple modules/files

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

# TODO: More elegant random seed handling
RANDOMSEED = 0

models = {
    'classification': {
        'LinearSVC': LinearSVC,
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC': SVC(),
        'GaussianNB': GaussianNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(random_state=RANDOMSEED),
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state=RANDOMSEED),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=RANDOMSEED),
        'LogisticRegression': LogisticRegression(),        
    },
    'regression': {
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'Ridge': Ridge(),
        'SVR': SVR(),
        'ExtraTreesRegressor': ExtraTreesRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor()
    }
}

# TODO: Add param grids for other models
param_grids = {
    'GradientBoostingClassifier': {
        'max_features': [0.6, 0.7, 0.8, 0.9],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [10, 12, 18, 25, 30],
        'n_estimators': [100, 150, 200]
    },
    'ExtraTreesClassifier': {
        'max_features': [0.6, 0.7, 0.8, 0.9],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split':[10, 12, 18, 25, 30],
        'n_estimators': [100, 150, 200]
    },
    'Ridge': {
        'alpha': [0.03,0.1, 3, 10, 30, 100]
    },
    'ExtraTreesRegressor': {
        'max_features': ['sqrt', 0.4, 0.8, 0.9],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [2, 5, 10, 20,],
        'n_estimators': [100, 150]
    },
    'RandomForestRegressor': {
        'max_features': ['sqrt', 0.4, 0.8, 0.9],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [2, 5, 10, 20,],
        'n_estimators': [100, 150]
    }
}

# TODO: Make compatible with regression.
def evaluate_models(X_train, y_train, prediction_type='classification', metric='accuracy'):
    ''' Evaluates models.
    
    takes a first pass at evaluating model performance with standards parameters.
    '''
    # Evaluate each models score
    results = []
    for name, model in models[prediction_type].items():
        kfold = model_selection.KFold(n_splits=5, random_state=RANDOMSEED)
        cv_results= model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=metric)
        results.append([name, cv_results.mean(), cv_results.std()])
    results_df = pd.DataFrame(results, columns=['model', 'score', 'std'])
    results_df.sort_values(by='score', ascending=False, inplace=True)
    return(results_df)

def model(X_train, y_train, prediction_type='classification', metric='accuracy'):
    ''' Returns a model ready to be trained
    '''
    models_evaled = evaluate_models(X_train, y_train, prediction_type, metric)
    best_model = models_evaled.ix[models_evaled['score'].idxmax()]['model']
    model = GridSearchCV(estimator=models[prediction_type][best_model], param_grid=param_grids[best_model], cv=5)
    return(model)
