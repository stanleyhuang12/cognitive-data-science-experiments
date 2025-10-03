from tools_rec import * 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from itertools import product

def compute_true_residuals(y, y_hat): 
    return (y - y_hat) ** 2

def compute_smoothed_residuals(y, baseline_y, alg_y): 
    return (
        ((y - baseline_y) ** 2) + 
        (2 * (y - baseline_y) * (alg_y - y)) + # Covariance term 
        ((alg_y - y) ** 2)
)

def compute_diff(y, y_hat):
    return (y - y_hat)

def create_interaction_terms(df, cat1, cat2, prefix): 
    new_df = df.copy()
    for col1 in cat1: 
        for col2 in cat2: 
            interaction_name = f"{prefix}_{col1}_x_{col2}"
            new_df[interaction_name] = new_df[col1] * new_df[col2]
            
    return new_df

def fit_new_linear_reg_model(df): 
    scaler = StandardScaler()
    new_df = df.copy()
    X = new_df.drop(['CC12'], axis=1).values
    X = scaler.fit_transform(X)
    y = new_df['CC12'].values - 1
    
    lr = LinearRegression()
    lr.fit(X, y)
    y_predictions = lr.predict(X)
    print('R_squared of new model:', lr.score(X, y))
    return lr.coef_, y_predictions

def compare_residuals_by_axes(df, residuals, axes, mapping=None): 
    
    out_df = pd.DataFrame(columns=['sum', 'n', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    
    for cat in sorted(df[axes].unique()): 
        print(cat)
        filtered_df = df[df[axes] == cat]
        
        descriptive_stats = filtered_df[residuals].describe()

        out_df_row = {
            'sum': filtered_df[residuals].sum(),
            'n': descriptive_stats['count'],
            'mean': descriptive_stats['mean'],
            'std': descriptive_stats['std'],
            'min': descriptive_stats['min'],
            '25%': descriptive_stats['25%'],
            '50%': descriptive_stats['50%'],
            '75%':  descriptive_stats['75%'],
            'max':  descriptive_stats['max']
        }
        print(out_df_row)
        
        if mapping: 
            name = mapping.get(cat, 'unknown_label')
            out_df.loc[name] = out_df_row
        else: 
            out_df.loc[cat] = out_df_row
        
        out_df = out_df.map(lambda x: np.round(x, 4))
        
    return out_df 
        

def lookup_features_from_top_k_residuals(df, feature_eval, residuals, k): 
    df[residuals] = df[residuals].map(lambda x: np.abs(x))
    data = df.sort_values(by=residuals, ascending=False, inplace=False)
    filtered_df = data[feature_eval][:k]

    descriptive_stats = filtered_df.describe()
    descriptive_stats.map(lambda x: np.round(x, 4))
    return descriptive_stats


def compute_regret_mass_for_binary_features(df, smoothed_resids_col, bundled_feature=None, verbose=True, column_maps=None): 
    
    """
    Description: 
    Regret mass is the average smoothed residuals of observations in a group G * proportion of observations that fall in group G.
    
    Args: 
    - df: A wide pandas.Dataframe of the dataset. Binary variables are disaggregated to multiple columns. 
    - smoothed_resids_cols: Points to the column with smoothed residuals. 
    - group_columns (default: None): if None, it uses all other columns as groups, otherwise specify the list of columns 
    """
    df = df.copy()
    
    if not bundled_feature: 
        group_cols = df.columns 
        group_cols.remove(smoothed_resids_col)
    else: 
        group_cols = bundled_feature
    
    if column_maps: 
        df = df.map(column_maps)
        
    total_obs = len(df)
    if verbose: 
        print("Total observations: ", total_obs)
    
    ret_df = pd.DataFrame(columns=group_cols, index=['class_proportion', 'average_residuals', 'regret_mass'])
    
    for col in group_cols: 
        class_proportion = df[col].sum() / total_obs
        ret_df.loc['class_proportion', col] = class_proportion
        
        average_residuals = np.mean(df[col] * df[smoothed_resids_col])
        ret_df.loc["average_residuals", col] = average_residuals
        
        regret_mass = class_proportion * average_residuals * 100
        ret_df.loc["regret_mass", col] = regret_mass
        
        if verbose: 
            print("=============================")
            print(f"Total observations in Group {col}", df[col].sum())
            print("Class proportion:", class_proportion)
            print("Average residuals: ", average_residuals)
            print("Regret mass for group", regret_mass)
        
    return ret_df.T


def compute_regret_mass_for_k_features(df: pd.DataFrame, 
                                       smoothed_residuals_col: str, 
                                       feature_dict: dict, 
                                       verbose=True,
                                       columns_map=None) -> pd.DataFrame: 
    """
    Description: 
    Regret mass for K binary features (non-overlapping)
    
    Args: 
    - df: A wide pandas.Dataframe of the dataset. Binary variables are disaggregated to multiple columns.
    - smoothed_residuals_col: Points to the column with smoothed residuals.
    - feature_dict: A dictionary of bundles of features. 
    
    Returns: 
    - A dataframe of class proportion, average residuals, regret mass weighted by class proportion and K features
    
    Example of feature_dict: 
    `gender` and `age` are names that will be set for the returned dataframe. 
        {
            "gender": "female", "male", "other", # column names 
            "age": "child", "teen", "adult", "senior", # column names 
        }
    """
    
    df = df.copy()
    if columns_map: 
        df = df.rename(columns_map, axis=1)
        mapped_feature_dict = {
            key: [columns_map.get(item, item) for item in values]
            for key, values in feature_dict.items()
        }
        feature_dict = mapped_feature_dict 

    
    ### Creates a pairwise list of interaction combinations 
    combinations = list(product(*feature_dict.values()))
    interaction_combinations = [list(tup) for tup in combinations]
    
    total_obs = len(df)
    if verbose: 
        print("Total observations: ", total_obs)
    ret_df = pd.DataFrame(columns=['class_proportion', 'average_residuals', 'regret_mass'])
    
    for c in interaction_combinations: 
        index_name = " x ".join(c)
    
        class_obs = (df[c] == 1).all(axis=1).sum()
        class_proportion = class_obs / total_obs
        ret_df.loc[index_name, "class_proportion"] = class_proportion

        matrix = df[c + [smoothed_residuals_col]].values
        smoothed_residuals = np.prod(matrix, axis=1)
        average_residuals = np.mean(smoothed_residuals)
        ret_df.loc[index_name, "average_residuals"] = average_residuals
        
        regret_mass = average_residuals * class_proportion * 100 
        ret_df.loc[index_name, "regret_mass"] = regret_mass
        
        if verbose: 
            print("=============================")
            print(f"Total observations in Group {index_name}", class_obs)
            print("Class proportion:", class_proportion)
            print("Average residuals: ", average_residuals)
            print("Regret mass for group", regret_mass)
    
    ret_df['normalized_regret_mass'] = ret_df['regret_mass'] / ret_df['regret_mass'].sum()
    
    return ret_df

def compare_top_n_smoothed_residuals(df, 
                                     smoothed_residuals, 
                                     top_N, 
                                     feature_dict, 
                                     **kwargs) -> tuple:
    """Compares `top_N` smoothed residuals' regret mass distribution to the full dataset. Returns a data that describes class proportion, 
    average residuals, normalized and unnormalized regret mass for each class and a 
    Jensen-Shannon divergence score for smoothed residual distribution. 
    
    Inputs: 
    
    - A dataframe 
    - A dataframe of the top N smoothed residuals 
    
    Computes `regret mass` and `normalized regret mass` for all features for both full dataset and top N smoothed residuals dataset
    
    Returns: 
    - A dataset of class proportions, average smoothed residuals, normalized and raw regret mass scores for each class. 
    - A Jensen-Shannon divergence score to compare distributions of smoothed residuas as evidence of latent classes. 
    """
    
    
    
        
        
       
        


greet = {"hi": ["hello", "bonjour"], 
 "bye": ["ciao", "byee", "wave"]}


list(product(*greet.values()))
    
    

    
    



def compute_regret_concentration_for_binary_features(): 
    pass
        
        


        
        
        
    

        
        
    