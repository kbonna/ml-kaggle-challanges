# modify this file and submit PR
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