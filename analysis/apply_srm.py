from scientific_regret import * 
from tools_rec import * 
from ray import tune
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if you don’t need to display the plot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from torcheval.metrics import R2Score
from sklearn.linear_model import LinearRegression
import math
from plot_graphs import plot_nn_training_validation_loss

data_wo_county = pd.read_csv('/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/data/CCES2012_CSVFormat_NEW_WC.csv')


search_space_init_configs = {
    'num_layers': tune.randint(1, 8),
    'base_exp': tune.randint(3, 9),
    'lr': tune.loguniform(1e-5, 1e-2),
}
    
train_configs_without_geocodes = {
    'wo_county': 1, 
    'file_path': '/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/data/CCES2012_CSVFormat_NEW_WC.csv',
    'input_dims': 32,
    'output_dims': 1, 
    'max_epochs': 100,
    'batch_size': 16384, 
    'activation_fn': 0, # Overwrite and force the model to perform tanh using a symmetric and scaled value 
}
torch.manual_seed(123)
def new_train_wo_geocodes(config): 
    
    # Train configs is a dictionary of items that does not vary 
    merged_configs = {**train_configs_without_geocodes, **config}
    dataset = load_dataset(merged_configs['file_path'])
    
    X = dataset.drop(['CC12'], axis=1).values.astype(np.float32)
    y = dataset['CC12'].values.reshape(-1, 1).astype(np.float32) - 1 ## Adjusted y axis 
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    train_idx, valid_idx = next(iter(kf.split(X)))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[valid_idx], y[valid_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
        
    # Convert to tensors 
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)
    
    model = initialize_model(merged_configs)
    
    # model = train.torch.prepare_model(model)
    
    criterion = nn.MSELoss()
    metric = R2Score()
    optimizer = optim.Adam(model.parameters(), lr=merged_configs['lr'])
    
    tensor_dataset = TensorDataset(X_train, y_train)
    
    dataloader = DataLoader(tensor_dataset, batch_size=merged_configs['batch_size'], num_workers=4, shuffle=True)
    r2_score_tracker = float('-inf')
    for epoch in range(merged_configs['max_epochs']):
        model.train()
        train_loss = 0 
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Compute model R2 
        valid_loss = 0
        model.eval()
        with torch.no_grad(): 
            y_pred = model.forward(X_test)
            valid_loss = criterion(y_pred, y_test)
            valid_loss = valid_loss.item()

            
        averaged_train_loss = train_loss / len(dataloader) # Average training loss per epoch 
        metric.reset()
        metric.update(y_pred, y_test)
        model_r2 = metric.compute().item()
        
        print(f"Epoch {epoch}: Current R2 = {model_r2}, Best R2 = {r2_score_tracker}")
        
        if model_r2 > r2_score_tracker: 
            ## Keep track of best model r2 score and save it as a checkpoint to call or continue from 
            r2_score_tracker = model_r2 
            
            metrics = {
            "step": epoch,
            "train_loss": averaged_train_loss,
            "valid_loss": valid_loss,
            "r2_score": model_r2,
            "best_r2_score": r2_score_tracker
            }
            
            with tempfile.TemporaryDirectory() as tmpdir: 
            ### REVISSE FOR PERSISTENT SˇORAGE PERSISTING CHECKPOINTS 
                checkpoint_to_track = None
                torch.save(
                    model.state_dict(),
                    os.path.join(tmpdir, "model.pth")
                )
                checkpoint_to_track = Checkpoint.from_directory(tmpdir)
                
                tune.report(metrics, checkpoint=checkpoint_to_track)
        
        else: 
            metrics = {
            "step": epoch,
            "train_loss": averaged_train_loss,
            "valid_loss": valid_loss,
            "r2_score": model_r2,
            "best_r2_score": r2_score_tracker
            }
            tune.report(metrics)
        
        print(f'Epoch {epoch}: Training loss {averaged_train_loss}, Validation loss {valid_loss}, R2 Score {model_r2}')

exp_path_wo_county = "/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints/env_pred_algo_run_without_county"
results_wo_county = tune.Tuner.restore('/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints/env_pred_algo_run_without_county', new_train_wo_geocodes)
result_grid_wo_county = results_wo_county.get_results()
best_results = result_grid_wo_county.get_best_result(metric='best_r2_score', mode='max')


load_data = torch.load(os.path.join(best_results.checkpoint.path, "model.pth"))

merged_configs = {**train_configs_without_geocodes, **best_results.config}
model = initialize_model(merged_configs)
nn_model = model.load_state_dict(load_data)


df_wo_codes = pd.read_csv(train_configs_without_geocodes['file_path'])
df_w_codes = pd.read_csv("data/CCES2012_CSVFormat_NEW.csv")
X = df_w_codes.drop(['CC12'], axis=1).values
y = df_w_codes['CC12'].values - 1

sc = StandardScaler()
lr = LinearRegression()
another_reg = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_idx, valid_idx = next(iter(kf.split(X)))
len(train_idx)
len(valid_idx)
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[valid_idx], y[valid_idx]

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
lr.score(X_test, y_test)
r2_score(y_test, y_pred)

another_reg.fit(X, y)
another_reg.score(X, y)


lr.fit(X, y)
y_pred_lr = lr.predict(X)
print('R2 for Linear Model:', lr.score(X, y))

plot_nn_training_validation_loss(X, y, intervals=10, configs=merged_configs)


model.eval()
with torch.no_grad(): 
    X_fit_t = torch.from_numpy(X_fit).float()
    y_t = torch.from_numpy(y).reshape(-1, 1).float()
    y_pred_t = model(X_fit_t)
    r2_scorer = R2Score()
    r2_scorer.update(y_pred_t, y_t)
    r2_score = r2_scorer.compute()
    print(f'R2 score: {r2_score}')
    
y = y.reshape(-1, 1)
y_pred_lr = y_pred_lr.reshape(-1, 1)
y_pred_mlp = y_pred_t.cpu().numpy().reshape(-1, 1)
exp_1_resids = pd.DataFrame(np.hstack((y, y_pred_lr, y_pred_mlp)), columns=['y_actual', 'y_pred_lr', 'y_pred_mlp'])



"""Start of SRM algorithm and comparing residuals"""


exp_1_resids['lr_resids'] = compute_true_residuals(exp_1_resids['y_actual'], exp_1_resids['y_pred_lr'])
exp_1_resids['mlp_resids'] = compute_true_residuals(exp_1_resids['y_actual'], exp_1_resids['y_pred_mlp'])

exp_1_resids['smoothed_resids'] = compute_smoothed_residuals(y=exp_1_resids['y_actual'], 
                                                             baseline_y=exp_1_resids['y_pred_lr'],
                                                             alg_y=exp_1_resids['y_pred_mlp'])
exp_1_resids['lr_pred_diff'] = exp_1_resids['y_actual']-exp_1_resids['y_pred_lr']
exp_1_resids['mlp_pred_diff'] = exp_1_resids['y_actual']-exp_1_resids['y_pred_mlp']

exp_1_resids = data_wo_county.merge(exp_1_resids, left_index=True, right_index=True)

top_100_smoothed_resids = exp_1_resids.sort_values(by='smoothed_resids', ascending=False).reset_index(drop=True).loc[:100]


age_columns = ["aged1", "aged2", "aged3", "aged5", "aged6"]
age_columns_name = ['18-24', '25-34', '35-44', '55-64', '65+']
race_columns = ["raced2", "raced3", "raced4"]
race_column_names = ['Black', 'Hispanic', 'Other']
ideology_columns = ['ideologyd1', 'ideologyd2', 'ideologyd4', 'ideologyd5']
ideology_columns_names = ['Very liberal', 'Liberal', 'Conservative', 'Very Conservative']
edu_columns = ['edu1', 'edu3', 'edu4']
edu_columns_names = ['High school\nor less',  'College Grad', 'Post-grad']
party_columns = ['PDd1', 'PDd2', 'PDd3', 'PDd5', 'PDd6', 'PDd7']
party_columns_name = ['Strong\nDem', 'Dem', 'Lean\nDem', 'Lean\nRep', 'Rep', 'Strong\nRep']
temp_columns = ['ccvar1_30_10','ccvar2_30_10', 'ccvar3_30_10', 'ccvar4_30_10']
temp_columns_name = [
        'Recent warming in\nstrong cool places',
        'Recent warming in\ncool places',
        #'No change',
        'Recent cooling in\nwarm places',
        'Recent cooling in\nstrong warm places'
    ]
label_map = {
    'age': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
    'gender': ['Female', 'Male'],
    'race': ['White', 'Black', 'Hispanic', 'Other'],
    'political ideology': ['Very\nLib', 'Lib', 'Moderate', 'Cons', 'Very\nCons'],
    'temp': [
        'Recent warming in\nstrong cool places',
        'Recent warming in\ncool places',
        'No change',
        'Recent cooling in\nwarm places',
        'Recent cooling in\nstrong warm places'
    ],
    'party': ['Strong\nDem', 'Dem', 'Lean\nDem', 'Ind', 'Lean\nRep', 'Rep', 'Strong\nRep']
}
def plot_srm_critique_intervention(data, columns, var, category_names, interact_cols=None, additional_desc="", k_smoothed_resids=100): 
    df = data.copy()
    
    if interact_cols: 
        resids_cols = ['y_actual', 'y_pred_lr', 'y_pred_mlp', 'lr_resids', 'mlp_resids', 'smoothed_resids', 'lr_pred_diff', 'mlp_pred_diff']
        df_with_interactions = create_interaction_terms(df.drop(columns=resids_cols, axis=1), columns, interact_cols, prefix="int")
        _, df_w_int_preds = fit_new_linear_reg_model(df_with_interactions)
        df['intervened_diff'] = df['y_actual'] - df_w_int_preds

    df = df.sort_values(by='smoothed_resids', ascending=False).reset_index(drop=True).loc[:k_smoothed_resids-1]
    df['category_index'] = np.argmax(df[columns].values, axis=1)
    
    plot_df = df[['lr_pred_diff', 'mlp_pred_diff'] + (['intervened_diff'] if interact_cols else [])].copy()
    plot_df['category_index'] = df['category_index']
    plot_df = plot_df.groupby('category_index').mean()
    
    full_class_indices = list(range(len(columns)))
    plot_df = plot_df.reindex(full_class_indices).fillna(0)
    print(plot_df)
    
    plot_df.index = category_names 

    plt.bar(x=plot_df.index, height=plot_df['lr_pred_diff'], color='lightgreen', edgecolor='black', label='Baseline model')
    if interact_cols: 
        plt.bar(x=plot_df.index, height=plot_df['intervened_diff'], color='mediumseagreen', edgecolor='black', label='Baseline with SRM critique')
    
    plt.bar(x=plot_df.index, height=plot_df['mlp_pred_diff'], color='darkgreen', edgecolor='black', label='Neural network model')

    plt.title(f"Mean absolute error for models stratified by {var} for top {k_smoothed_resids} observations with highest smoothed residuals\n{additional_desc}", color='black')
    plt.axhline(y=0, color='red')
    plt.xlabel(var[0].upper() + var[1:])
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()


plot_srm_critique_intervention(exp_1_resids,
                               columns=age_columns, 
                               var="age", 
                               category_names=age_columns_name, 
                               interact_cols=ideology_columns,
                               additional_desc="Baseline under-predicts for younger and over-predicts for older people\nAugmented model includes interaction with age and political ideology")

plot_srm_critique_intervention(exp_1_resids,
                               columns=ideology_columns, 
                               var='political ideology',
                               category_names=ideology_columns_names,
                               interact_cols=race_columns,
                               additional_desc="Augmented model adjusted for interaction between race and political ideology")

plot_srm_critique_intervention(exp_1_resids,
                               columns=party_columns,
                               var="party affiliation",
                               category_names=party_columns_name,
                               interact_cols=race_columns, 
                               additional_desc="Augmented model adjusted for interaction between race and party affiliation")

plot_srm_critique_intervention(exp_1_resids, 
                               columns=ideology_columns, 
                               var="ideology",
                               category_names=ideology_columns_names, 
                               interact_cols=edu_columns,
                               additional_desc="Augmented model adjusted for interaction between ideology and education")
np.unique(np.argmax(exp_1_resids[party_columns], axis=1))
def plot_side_by_side_histograms(data1, data2, category_labels, title1="Group 1", title2="Group 2"):
    """
    Plots two histograms side-by-side with discrete category labels.

    Args:
        data1 (array-like): First dataset (e.g., top 100 residuals).
        data2 (array-like): Second dataset (e.g., full population).
        category_labels (list): Labels corresponding to the categories. Must match number of unique categories.
        title1 (str): Title for the left histogram.
        title2 (str): Title for the right histogram.
    """
    num_categories = len(category_labels)
    bins = np.arange(num_categories + 1) - 0.5  # Center bins

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Left plot
    axes[0].hist(data1, bins=bins, edgecolor='black', color='salmon')
    axes[0].set_title(title1)
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(np.arange(num_categories))
    axes[0].set_xticklabels(category_labels, rotation=45, ha='right')

    # Right plot
    axes[1].hist(data2, bins=bins, edgecolor='black', color='lightblue')
    axes[1].set_title(title2)
    axes[1].set_xticks(np.arange(num_categories))
    axes[1].set_xticklabels(category_labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    
    
def convert_to_continuous_variable(df, columns, base_index): 
    all_zero_mask = (df[columns] == 0).all(axis=1)
    base_vec = np.where(all_zero_mask, 1, 0).reshape(-1, 1)
    other_vals = df[columns].values
    n_cols = len(columns)
    
    if base_index == 0:
        combined = np.hstack((base_vec, other_vals))
    elif base_index == n_cols:
        combined = np.hstack((other_vals, base_vec))
    else:
        combined = np.hstack((
            other_vals[:, :base_index],
            base_vec,
            other_vals[:, base_index:]
        ))
    
    categorical = np.argmax(combined, axis=1)
    return categorical.reshape(-1,1)


def plot_over_under_by_categories(df, category_cols, y_pred_col, y_actual_col):
    df = df.copy()
    
    # Define over/under prediction
    df['residual'] = df[y_pred_col] - df[y_actual_col]
    df['prediction_type'] = df['residual'].apply(lambda x: 'Overpredicted' if x > 0 else 'Underpredicted')

    for col in category_cols:
        if df[col].nunique() > 20:
            print(f"Skipping {col} — too many unique values.")
            continue

        # Group and count
        counts = df.groupby([col, 'prediction_type']).size().unstack(fill_value=0)

        # Sort categories for consistent display
        counts = counts.sort_index()

        # Plot
        ax = counts.plot(kind='bar', 
                         stacked=True,
                         color=['lightblue', 'salmon'], 
                         edgecolor='black',
                         figsize=(8, 5))
        
        plt.title(f'Over vs Under Predictions by {col}')
        plt.ylabel('Count')
        plt.xlabel(col)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Prediction Type')
        plt.tight_layout()
        plt.show()
        

def plot_over_under_subplots(df, category_cols, y_pred_col, y_actual_col, label_map=None):
    df = df.copy()
    df['residual'] = df[y_pred_col] - df[y_actual_col]
    df['prediction_type'] = df['residual'].apply(lambda x: 'Overpredicted' if x > 0 else 'Underpredicted')

    n = len(category_cols)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5))
    axes = axes.flatten()

    handles, labels = None, None

    for i, col in enumerate(category_cols):
        if df[col].nunique() > 20:
            print(f"Skipping '{col}' — too many unique values.")
            continue

        ax = axes[i]

        
        for col in category_cols:
            if label_map and col in label_map:
                df[col] = df[col].map(lambda x: label_map[col][x])

        # Group and sort
        counts = df.groupby([col, 'prediction_type']).size().unstack(fill_value=0)

        # Plot
        counts.plot(kind='bar', 
                    color=['lightblue', 'salmon'], 
                    edgecolor='black',
                    ax=ax,
                    legend=False)

        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()

        ax.set_title(f'Over vs Under by {col}', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.legend(handles, labels, loc='upper center', ncol=2, title='Prediction Type', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



plot_disagg_df = pd.DataFrame(index=np.arange(0, 100, 1))
plot_disagg_df['y_pred_lr'] = top_100_smoothed_resids[:100]['y_pred_lr']
plot_disagg_df['y_pred_actual'] = top_100_smoothed_resids[:100]['y_actual']
plot_disagg_df['age'] = convert_to_continuous_variable(top_100_smoothed_resids[:100], age_columns, 3)
plot_disagg_df['gender'] = convert_to_continuous_variable(top_100_smoothed_resids[:100], ['Gender'], 0)
plot_disagg_df['race'] = convert_to_continuous_variable(top_100_smoothed_resids[:100], race_columns, base_index=0)
plot_disagg_df['ideology'] = convert_to_continuous_variable(top_100_smoothed_resids[:100], ideology_columns, base_index=2)
plot_disagg_df['temp'] = convert_to_continuous_variable(top_100_smoothed_resids[:100], temp_columns, 2)
plot_disagg_df['party']  = convert_to_continuous_variable(top_100_smoothed_resids[:100], party_columns, 2)
label_map = {
    'age': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
    'gender': ['Female', 'Male'],
    'race': ['White', 'Black', 'Hispanic', 'Other'],
    'ideology': ['Very\nLib', 'Lib', 'Moderate', 'Cons', 'Very\nCons'],
    'temp': [
        'Recent warming in\nstrong cool places',
        'Recent warming in\ncool places',
        'No change',
        'Recent cooling in\nwarm places',
        'Recent cooling in\nstrong warm places'
    ],
    'party': ['Strong\nDem', 'Dem', 'Lean\nDem', 'Ind', 'Lean\nRep', 'Rep', 'Strong\nRep']
}

plot_over_under_subplots(
    plot_disagg_df,
    category_cols=['age', 'gender', 'race', 'ideology', 'temp', 'party'],
    y_pred_col='y_pred_lr',
    y_actual_col='y_pred_actual',
)

label_map = {
    # Race
    'raced2': 'Black',
    'raced3': 'Hispanic',
    'raced4': 'Other',

    # Gender
    'Gender': 'Male',

    # Ideology
    'ideologyd1': 'Very\nLib',
    'ideologyd2': 'Lib',
    'ideologyd4': 'Cons',
    'ideologyd5': 'Very\nCons',

    # Education
    'edu1': 'High School\nor less',
    'edu3': 'College\ngrad',
    'edu4': 'Post-grad',

    # Party ID
    'PDd1': 'Strong\nDem',
    'PDd2': 'Dem',
    'PDd3': 'Lean\nDem',
    'PDd5': 'Ind',
    'PDd6': 'Lean\nRep',
    'PDd7': 'Rep',

    # Age
    'aged1': '18-24',
    'aged2': '25-34',
    'aged3': '35-44',
    'aged5': '55-64',
    'aged6': '65+',

    # Temperature Change
    'ccvar1_30_10': 'Recent warming in\nstrong cool places',
    'ccvar2_30_10': 'Recent warming in\ncool places',
    'ccvar3_30_10': 'No change',
    'ccvar4_30_10': 'Recent cooling in\nwarm places',
}

def compare_all_categorical_variables_one_figure(df_subset, df_full, variable_groups):
    """
    Plots multiple categorical comparisons in a single unified figure (multi-row layout).

    Args:
        df_subset (pd.DataFrame): Filtered subset.
        df_full (pd.DataFrame): Full dataset.
        variable_groups (list of dict): Each dict must include:
            - 'columns': list of one-hot column names
            - 'title': str, name for the plot
            - 'labels': list of str, labels including base
    """
    n = len(variable_groups)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 4 * n))

    for i, group in enumerate(variable_groups):
        columns = group['columns']
        title = group['title']
        labels = group['labels']
        base_index = group['base_index']
        num_categories = len(labels)
        bins = np.arange(num_categories + 1) - 0.5

        # Convert one-hot to continuous
        cats_subset = convert_to_continuous_variable(df_subset, columns, base_index=base_index)
        cats_full = convert_to_continuous_variable(df_full, columns, base_index=base_index)

        # Subset plot (left)
        axes[i, 0].hist(cats_subset, bins=bins, edgecolor='black', color='skyblue')
        axes[i, 0].set_title(f"{title} (Subset)")
        axes[i, 0].set_ylabel("Count")
        axes[i, 0].set_xticks(range(num_categories))
        axes[i, 0].set_xticklabels(labels, rotation=45, ha='right')

        # Full plot (right)
        axes[i, 1].hist(cats_full, bins=bins, edgecolor='black', color='salmon')
        axes[i, 1].set_title(f"{title} (All Data)")
        axes[i, 1].set_xticks(range(num_categories))
        axes[i, 1].set_xticklabels(labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    
    
variable_groups = [
    {
        'columns': ['raced2', 'raced3', 'raced4'],  # race dummies
        'title': 'Race',
        'labels': ['White', 'Black', 'Hispanic', 'Other'],
        'base_index': 0
    },
    {
        'columns': ['aged1', 'aged2', 'aged3', 'aged5', 'aged6'],  # age dummies
        'title': 'Age',
        'labels': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        'base_index': 3
    },
    {
        'columns': ['ideologyd1', 'ideologyd2', 'ideologyd4', 'ideologyd5'],  # ideology dummies
        'title': 'Political Ideology',
        'labels': ['Very\nLib', 'Lib', 'Moderate', 'Cons', 'Very\nCons'],
        'base_index': 2
    },
   {
        'columns': ['PDd1', 'PDd2', 'PDd3', 'PDd5', 'PDd6', 'PDd7'],  # party dummies
        'title': 'Party identification',
        'labels': ['Strong\nDem', 'Dem', 'Lean\nDem', 'Ind', 'Lean\nRep', 'Rep', 'Strong\nRep'],
        'base_index': 3
    }
]

variable_groups_2 = [
    {
        'columns': ['PDd1', 'PDd2', 'PDd3', 'PDd5', 'PDd6', 'PDd7'],  # party dummies
        'title': 'Party identification',
        'labels': ['Strong\nDem', 'Dem', 'Lean\nDem', 'Ind', 'Lean\nRep', 'Rep', 'Strong\nRep'],
        'base_index': 3
    },
    {
        'columns': ['aged1', 'aged2', 'aged3', 'aged5', 'aged6'],  # age dummies
        'title': 'Age',
        'labels': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        'base_index': 3
    },
    {
        'columns': ['ccvar1_30_10','ccvar2_30_10', 'ccvar3_30_10', 'ccvar4_30_10'],  # recent temp changes
        'title': 'Recent temperature changes',
        'labels': ['Recent warming in\nstrong cool places', 
                   'Recent warming in\ncool places',
                   'No change',
                   'Recent cooling in\nwarm places', 
                   'Recent cooling in\nstrong warm places'],
        'base_index': 2
    }
]


compare_all_categorical_variables_one_figure(df_subset=top_100_smoothed_resids, 
                                             df_full=exp_1_resids, 
                                             variable_groups=variable_groups)

compare_all_categorical_variables_one_figure(df_subset=top_100_smoothed_resids,
                                             df_full=exp_1_resids, 
                                             variable_groups=variable_groups_2)

np.where(exp_1_resids['ccvar4_30_10'] != 0, 1, 0).sum()
# plot_srm_critique_intervention(exp_1_resids, 
#                                columns=age_columns, 
#                                category_names=['18-24', '25-34', '35-44', '55-64', '65+'], 
#                                interact_cols=race_columns, 
#                                additional_desc=None)

plot_srm_critique_intervention(exp_1_resids, 
                               columns=age_columns,
                               var="age",
                               category_names=['18-24', '25-34', '35-44', '55-64', '65+'],
                               interact_cols=ideology_columns, 
                               additional_desc="Age interacts with ideology helps explain some of the variance")




all_zero_mask = (top_100_smoothed_resids[['raced2', 'raced3', 'raced4']] == 0).all(axis=1)


resids_cols = ['y_actual', 'y_pred_lr', 'y_pred_mlp', 'lr_resids', 'mlp_resids', 'smoothed_resids', 'lr_pred_diff', 'mlp_pred_diff']
relevant_cols = [col for col in top_100_smoothed_resids if col not in resids_cols]



race_cats = convert_to_continuous_variable(top_100_smoothed_resids, race_columns)
race_cats_total = convert_to_continuous_variable(df_wo_codes, race_columns)

plot_side_by_side_histograms(race_cats, 
                             race_cats_total, 
                             category_labels = ['White', 'Black', 'Hispanic', 'Not White, Black, or Hispanic'],
                             title1='Top 100 mismatched observations',
                             title2='Full sample', 
                             )

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

# Histogram for top_100_smoothed_resids
axes[0].hist(race_cats, bins=range(len(race_columns) + 2), edgecolor='black')
axes[0].set_title('Top 100 Smoothed Residuals')
axes[0].set_xlabel('Race Category')
axes[0].set_ylabel('Count')

# Histogram for full dataset
axes[1].hist(race_cats_total, bins=range(len(race_columns) + 2), edgecolor='black')
axes[1].set_title('All Observations')
axes[1].set_xlabel('Race Category')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

relevant_cols = [col for col in top_100_smoothed_resids if col not in resids_cols]





plt.bar()
top_100_smoothed_resids[race_columns]