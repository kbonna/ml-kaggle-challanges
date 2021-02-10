import pandas as pd
import phik
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import VarianceThreshold

def removeOutliers(data: pd.DataFrame, method: str, treshold: float):
     """ 
    Function try to find outliers in given data set using one of available methods: DBSSCAN, OneClassSVN, IsolationForest. 
    Next step is reduce number of outliers to given treshold. Where treshold is percentage of population.
    Value 0.05 means that function will return data set without 5% of population which were marked as outlier.
    If algorithm detected number of outliers less than treshold then funtion remove all detected outliers from population.
    
    Args:
        data (Pd.DataFrame): Data set which should contains only features.
        threshold (float): Value between 0 and 1.
        method (str): Method used to calculate corelation between columns. Available methods: DBSSCAN, SVN, IsolationForest
    Returns:
        Data Frame without observation marked as outliers.

    """
    
    methods = {'DBSSCAN':  DBSCAN(eps = 1_500),
         'SVN': OneClassSVM(kernel = 'linear', nu = .05),
         'IsolationForest': IsolationForest( behaviour = 'new', random_state = 1, contamination= treshold)}
    model = methods[method]
    outliers = model.fit_predict(data)
    
    lowerBound = np.floor(treshold * len(data))
    toRemove = data.loc[outliers==-1 ,:].index
    toRemove = toRemove[: int(min(lowerBound, len(toRemove)))]
    return data.drop(toRemove)

def removeHighlyCorreletedFeatures(data: pd.DataFrame, threshold: float, method: str):
    """ 
    Function calculates correlation between features and removes one of these features which are highly correleted.
    Warning: if two features are highly correleted then function take first feature from list data.columns.
    Highly correleted features are these which absolute value of correlation is bigger than treshold. 
    Args:
        data (Pd.DataFrame): Data set which should contains only features.
        threshold (float): Value between 0 and 1.
        method (str): Method used to calculate corelation between columns. Available methods: {‘pearson’, ‘kendall’, ‘spearman’} or callable
    Returns:
        Data Frame with not correleted features between each other.

    """
    
    
    unCorrelatedFeatures = set()
    
    corr_data = data1.corr(method=method)
    corr_data = corr_data[np.abs(corr_data) < threshold]
    unCorrelatedFeatures.add(corr_data.columns[0])
    columnsToCheck = corr_data[corr_data.columns[0]].dropna().index
    for column in columnsToCheck:
        previousColumns = corr_data.loc[corr_data.index.isin(unCorrelatedFeatures),column]
        if np.sum(previousColumns < threshold) == len(unCorrelatedFeatures):
            unCorrelatedFeatures.add(column)
    
    return data[unCorrelatedFeatures]

def remove_uncorrelated_with_target(data, target_column: str, 
                                    threshold: float = 0.3, 
                                    method: str = 'pearson', 
                                    remove_anticorrelated: bool = True,
                                    interval_cols: list = None,
                                    verbose: bool = False):
    '''
    This function removes features that are weakly correlated to the target column.
    
    Args:
        data (DataFrame): data to remove the uncorrelated columns from,
        
        target_column (str): target column to compare the other features with,
        
        threshold (float): correlation threshold,
        
        method (str): 'pearson', 'kendall', 'spearman', 'phik':
            - 'pearson' - standard correlation coefficient
            - 'kendall' - Kendall Tau correlation coefficient 
            - 'spearman' - Spearman rank correlation 
            - 'phik' - described at https://arxiv.org/abs/1811.11440)
        
        remove_anticorrelated (bool): determines if features that are anti-correlated
            above the given threshold should be removed from the dataframe (True, default) 
            or should be left in the data (False)
        
        interval_cols (list) - column names of columns with interval variables, only relevant
            in the 'phik' method, the default behaviour is to assume that all columns 
            are numerical
            
        verbose (bool) - this mode if turned on (True) provides details about the execution of the code
        
    Returns:
        data_copy (DataFrame): data after removing the weakly correlated features
        
        removed_columns (list): names of removed columns

     '''
    
    data_copy = data.copy()

    if method in ['pearson','kendall','spearman']:
        corr = data_copy.corr(method=method)
    elif method == 'phik':
        if interval_cols:
            corr = data_copy.phik_matrix(interval_cols=interval_cols)
        else:
            corr = data_copy.phik_matrix(interval_cols=data_copy.columns)
    else: 
        raise KeyError('Provide a valid method name.')
    
    # Show features with highest and lowest correlation with the target column
    corr_target = corr[target_column]
    corr_target = corr_target.sort_values(ascending=False)
    if verbose:
        print(f'Features with highest correlation with the target columns are: \n{corr_target.head(5)}')
        print(f'Features with lowest correlation with the target columns are: \n{corr_target.tail(5)}')

    # Remove features weakly correlating with target 
    features_to_remove = corr_target[corr_target < threshold].index
    if remove_anticorrelated:
         features_to_keep = corr_target[abs(corr_target) > threshold].index
    else:
         features_to_keep = corr_target[corr_target > threshold].index
            
    data_copy = data_copy[features_to_keep]
    if verbose:
        print(f'({len(features_to_remove)} features correlated with target column ({target_column})',
              f'weaker than {threshold} have been removed',
              f'and the data now has the shape {data_copy.shape}.')
    
    return data_copy, list(features_to_remove)

def remove_low_variance_features(data, threshold: float = 0.05):
    '''
    This function removes features with low variance.
    
    Arguments:
    
    data (DataFrame): data to remove the low-variance features from
    threshold (float): minimum variance that the feature needs to have in order to stay in the dataset
    
    Returns:
    
    data_high_variance (DataFrame): data with the low-variant features removed
    removed_cols (list): removed columns
    
    '''
    
    data_copy = data.copy()
    raw_variances = data_copy.var()
    means = data_copy.mean()
    # an attempt to normalize the variance: var/(mean**2) should be lower than the threshold
    cols_to_drop = [column for column in data.columns if raw_variances[column]/(means[column]**2) < threshold]
    data_high_variance = data_copy.drop(columns = cols_to_drop)
    
    return data_high_variance, cols_to_drop
