# modify this file and submit PR

def remove_uncorrelated_with_target(data, target_columns: str, 
                                    threshold: float = 0.3, 
                                    method: str = 'pearson', 
                                    remove_anticorrelated: bool = False):
    '''
    This function removes features that are weakly correlated to the target columns.
    
    Args:
        target_columns (str): columns to correlate the others with,
        
        threshold (float): correlation threshold,
        
        method (str): 'pearson', 'kendall', 'spearman', 'phik':
            - 'pearson' - standard correlation coefficient
            - 'kendall' - Kendall Tau correlation coefficient 
            - 'spearman' - Spearman rank correlation 
            - 'phik' - described at https://arxiv.org/abs/1811.11440), note
                that interval_cols will always be equal to all columns from 
                data since we assume that the data is provided after encoding
        
        remove_anticorrelated (bool): determines if features that are anti-correlated
            above the given threshold should be removed from the dataframe (True) 
            or should be left in the data (False, default)
    Returns:
        data_copy - data after removing the weakly correlated features
    '''
    
    data_copy = data.copy()

    if method in ['pearson','kendall','spearman']:
        corr = data_copy.corr(method=method)
    elif method == 'phik':
        corr = data_copy.phik_matrix(interval_cols = data_copy.columns)
    else: 
        print('Provide a valid method name.')

    ''' NOTE FOR LATER: It would be good to store the correlation matrix as an instance property
    when we will put it into the class. Then one can have access to the correlation matrix 
    later, e.g. to plot it and have a look on it. '''
    
    # Show features with highest and lowest correlations with target columns
    corr_target = corr[target_columns]
    corr_target = corr_target.sort_values(ascending=False)
    print(f'Features with highest correlation with the target columns are: \n{corr_target.head(5)}')
    print(f'Features with lowest correlation with the target columns are: \n{corr_target.tail(5)}')

    # Remove features that are weakly correlated with target 
    features_to_remove = corr_target[corr_target < threshold].index
    features_to_keep = corr_target[corr_target > threshold].index
    if not remove_anticorrelated:
         features_to_keep = corr_target[abs(corr_target) > threshold].index
    data_copy = data_copy[features_to_keep]
    print(f'({len(features_to_remove)} features correlated with target columns ({target_columns})',
          f'weaker than {threshold} have been removed',
          f'and the data now has the shape {data_copy.shape}.')
    
    return data_copy
