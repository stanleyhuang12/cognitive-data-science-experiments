from tools_rec import *
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ray import train
from ray import tune
# from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from sklearn.model_selection import KFold, train_test_split
from ray import tune
from ray.tune.schedulers import ASHAScheduler # Scheduler 
from torcheval.metrics import R2Score 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from plot_graphs import * 
from ray.tune.search import hyperopt #Hyperopt performs tree-structured parzen estimator
from scientific_regret import * 

"=====================Bayesian hyperparameter search====================="


data_file_path_geocoded = '/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/data/CCES2012_CSVFormat_NEW_GEOCODED.csv'
data_file_path_without_county = '/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/data/CCES2012_CSVFormat_NEW_WC.csv'


search_space_init_configs = {
    'num_layers': tune.randint(1, 8),
    'base_exp': tune.randint(3, 9),
    'lr': tune.loguniform(1e-5, 1e-2),
}

train_configs = {
    'file_path': data_file_path_geocoded,
    'input_dims': 34,
    'output_dims': 1,
    'max_epochs': 100,
    'batch_size': 8192
}

def scale_y(vec): 
    vec -= 3 # CEnter the vector 
    vec *= 0.5 # Scale into smaller values (to avoid values getting close to 1)
    return np.tanh(vec)

def inverse_scale_y(vec): 
    vec = np.arctanh(vec)
    return (vec * 2) + 3 

    
train_configs_without_geocodes = {
    'wo_county': 1, 
    'file_path': data_file_path_without_county,
    'input_dims': 32,
    'output_dims': 1, 
    'max_epochs': 100,
    'batch_size': 16384, 
    'activation_fn': 0, # Overwrite and force the model to perform tanh using a symmetric and scaled value 
}
torch.manual_seed(123)
# os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

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


target_specs={
    'raced2': 0.7,
    'ideologyd4': 0.5,
    'ideologyd5': 0.3,
    'PDd1': 0.8,
    'edu1': 0.7,
    'aged5': 0.5
}
multivariate_stratified_sample = controlled_binary_sample(pd.read_csv(data_file_path_without_county), target_specs, sample_size=9000, seed=17)
multivariate_stratified_sample.to_csv('data/multivariate_strat_sample_1.csv', index=False)

pretrain_configs = {
    "file_path": "/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/data/multivariate_strat_sample_1.csv",
    "input_dims": 32,
    "activation_fn": 0,
    "base_exp": 7,
    "num_layers": 3,
    "input_dims": 32,
    "max_epoch": 100,
    "batch_size": 1000,
    "lr": 0.003
           }


pretrain_trainer= TorchTrainer(train_loop_per_worker=new_train_wo_geocodes, 
             train_loop_config=pretrain_configs,
             run_config=tune.RunConfig(name='pretrain_on_multivariate_sample',
                                       storage_path='/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints',
                                       checkpoint_config=tune.CheckpointConfig(num_to_keep=10,
                                                                               checkpoint_score_attribute='valid_loss', 
                                                                               checkpoint_score_order="min")),
             )
pretrain_results = pretrain_trainer.fit()
best_warm_start_path = pretrain_results.get_best_checkpoint(metric='best_r2_score', mode='max').path #Model we want 

resume_on_pretrain = {
    'pretrain': os.path.join(best_warm_start_path, "model.pth")
}

resume_tune_pretrain = {**resume_on_pretrain, **pretrain_configs, **train_configs_without_geocodes}


def warm_train_wo_geocodes(config): 
    
    # Train configs is a dictionary of items that does not vary 
    merged_configs = {**resume_tune_pretrain, **config}
    dataset = load_dataset(merged_configs['file_path'])
    
    X = dataset.drop(['CC12'], axis=1).values.astype(np.float32)
    y = dataset['CC12'].values.reshape(-1, 1).astype(np.float32) - 1 ## Adjusted y axis 
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    train_idx, valid_idx = next(iter(kf.split(X))) ## NOTE: Need to iterate over the splits 
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

warm_trainer = TorchTrainer(train_loop_per_worker=new_train_wo_geocodes, 
             train_loop_config=resume_tune_pretrain,
             run_config=tune.RunConfig(name='resume_on_pretrain',
                                       storage_path='/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints',
                                       checkpoint_config=tune.CheckpointConfig(num_to_keep=10,
                                                                               checkpoint_score_attribute='valid_loss', 
                                                                               checkpoint_score_order="min")),
             scaling_config=train.ScalingConfig(num_workers=1)
             )

warm_results = warm_trainer.fit()

def new_train_fn(config): 
    
    # Train configs is a dictionary of items that does not vary 
    merged_configs = {**train_configs, **config}
    dataset = load_dataset(merged_configs['file_path'])
    
    X = dataset.drop(['CC12'], axis=1).values.astype(np.float32)
    y = dataset['CC12'].values.reshape(-1, 1).astype(np.float32) - 1 ## Adjusted y axis 
    
    if merged_configs.get('scale_y', 'False'): 
        y = scale_y(y_train)
    
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

## average epoch is miscalculated 
    
# trainer = TorchTrainer(new_train_fn, train_loop_config=param_configs, 
#                        run_config=tune.RunConfig('/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints/train_practice',
#                                             checkpoint_config=tune.CheckpointConfig(num_to_keep=1, 
#                                                                                checkpoint_score_attribute='valid_loss'))
# )
 
# res = trainer.fit()


## Pretrain neural network 



# if there is a checkpoint configuration, should we use this to do something? 
asha_scheduler = ASHAScheduler(max_t=100,
                               grace_period=8,
                               reduction_factor=2, 
                               metric='valid_loss',
                               mode='min')

tree_parzen_estimator = hyperopt.HyperOptSearch(metric='valid_loss', 
                                                mode='min')

tuner = tune.Tuner(tune.with_resources(new_train_fn, {"cpu": 2}),  
                   param_space=search_space_init_configs, 
                   tune_config=tune.TuneConfig(num_samples=44,
                                          max_concurrent_trials=4, 
                                          scheduler=asha_scheduler, 
                                          search_alg=tree_parzen_estimator),
                   run_config=tune.RunConfig(name='env_pred_tpe_algo_run',
                                        storage_path='/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints',
                                        checkpoint_config=tune.CheckpointConfig(num_to_keep=10,
                                        checkpoint_score_attribute='valid_loss', 
                                        checkpoint_score_order="min"), 
                                        ), 
                   

                   )

results = tuner.fit()

tuner_wo_geocodes = tune.Tuner(tune.with_resources(new_train_wo_geocodes, {"cpu": 2}),  
                   param_space=search_space_init_configs, 
                   tune_config=tune.TuneConfig(num_samples=44,
                                          max_concurrent_trials=4, 
                                          scheduler=asha_scheduler, 
                                          search_alg=tree_parzen_estimator),
                   run_config=tune.RunConfig(name='env_pred_algo_run_without_county',
                                        storage_path='/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints',
                                        checkpoint_config=tune.CheckpointConfig(num_to_keep=10,
                                        checkpoint_score_attribute='valid_loss', 
                                        checkpoint_score_order="min"), 
                                        ), 
                   )

results_wo_county = tuner_wo_geocodes.fit()

warm_start_tuner_ = tune.Tuner(tune.with_resources(warm_train_wo_geocodes, {"cpu": 2}),  
                   param_space=search_space_init_configs, 
                   tune_config=tune.TuneConfig(num_samples=24,
                                          max_concurrent_trials=4, 
                                          scheduler=asha_scheduler, 
                                          search_alg=tree_parzen_estimator),
                   run_config=tune.RunConfig(name='warm_start_runs',
                                        storage_path='/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints',
                                        checkpoint_config=tune.CheckpointConfig(num_to_keep=5,
                                        checkpoint_score_attribute='valid_loss', 
                                        checkpoint_score_order="min"), 
                                        )
                   )

warm_results = warm_start_tuner_.fit()

torch.load(resume_tune_pretrain['pretrain'])
## Restore results woithout county data 
exp_path_wo_county = "/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints/env_pred_algo_run_without_county"

## Export experiment results 
results_wo_county = tune.Tuner.restore(exp_path_wo_county, new_train_wo_geocodes).get_results()
experiment_1 = results_wo_county.get_dataframe()
experiment_1.sort_values(by='best_r2_score', ascending=False, inplace=True)
experiment_1.to_csv('res/experiment_1.csv')

# Load best model 
best_results = results_wo_county.get_best_result(metric='best_r2_score', mode='max')
wo_county_path = best_results.checkpoint.path
merged_configs = {**train_configs_without_geocodes, **best_results.config}
model_data= torch.load(os.path.join(wo_county_path, "model.pth"))
model = initialize_model(merged_configs)
model.load_state_dict(model_data)



## Export experiment results with county dummies

results_w_county = tune.Tuner.restore("/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints/env_pred_tpe_algo_run", new_train_fn).get_results()
results_w_county.get_best_result(metric='best_r2_score', mode='max')

# Load data 
df_wo_codes = pd.read_csv(train_configs_without_geocodes['file_path'])
X = df_wo_codes.drop(['CC12'], axis=1).values
y = df_wo_codes['CC12'].values - 1

sc = StandardScaler()
X_fit = sc.fit_transform(X)
lr = LinearRegression()

lr.fit(X, y)
y_pred_lr = lr.predict(X)
print('R2 for Linear Model:', lr.score(X, y))

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


exp_1_resids['raw_resids'] = compute_true_residuals(y=exp_1_resids['y_actual'], y_hat=exp_1_resids['y_pred_lr'])
mask_ = np.where((exp_1_resids['raw_resids'] < exp_1_resids['smoothed_resids']), 1, 0)
exp_1_resids['sm_greater_than_rw'] = mask_
df_wo_geocodes = pd.concat([df_wo_codes, exp_1_resids], axis=1)

exp_1_resids.to_pickle("srm/green_policies_resids.pickle")

demographics_col = ["raced2", "raced3", "raced4", "Gender", "aged1", "aged2", "aged3", "aged5", "aged6", "edu1", "edu3", "edu4"]
df_for_poor_ml_model = df_wo_geocodes[(df_wo_geocodes['sm_greater_than_rw'] == 1)].loc[0:4000].reset_index(drop=True)

df_for_highest_resids = df_wo_geocodes.sort_values(by='smoothed_resids', ascending=False).loc[0:4000].reset_index(drop=True)


df_for_poor_ml_model[demographics_col].describe()
df_for_highest_resids[demographics_col].describe()











df_wo_geocodes.loc[(df_wo_geocodes['sm_greater_than_rw'] == 1), demographics_col].describe()


mlp_linear_diff = df_wo_geocodes['y_pred_mlp'] - df_wo_geocodes['y_pred_lr']
plt.hist(mlp_linear_diff, color='skyblue', edgecolor='black')
plt.axvline(x=0, color='red')
plt.title('Absolute difference between MLP and linear model predictions')
plt.show()

mlp_linear_resids_diff = df_wo_geocodes['lr_resids'] - df_wo_geocodes['mlp_resids']
plt.hist(mlp_linear_resids_diff, bins=10, color='skyblue', edgecolor='black')
plt.axvline(x=0, color='red')
plt.title('Absolute difference of residuals between MLP and linear model predictions')
plt.show() 

smoothed_resids_dist = df_wo_geocodes['smoothed_resids']
plt.hist(smoothed_resids_dist, color='cyan', edgecolor='black')
plt.axvline(x=0, color='red')
plt.title('Distribution of smoothed residuals')
plt.show()


### Stats 
df_wo_geocodes_gender = compare_residuals_by_axes(df=df_wo_geocodes, 
                                                  residuals='smoothed_resids',  
                                                  axes='Gender', 
                                                  mapping={0.0:'f', 1.0:'m'})

df_wo_geocodes_political = compare_residuals_by_axes(df=df_wo_geocodes, 
                                                     residuals='smoothed_resids', 
                                                     axes='ideologyd1', 
                                                     )

df_wo_geocodes_political_2 = compare_residuals_by_axes(df=df_wo_geocodes, 
                                                     residuals='smoothed_resids', 
                                                     axes='ideologyd2', 
                                                     )

df_wo_geocodes_political_2 = compare_residuals_by_axes(df=df_wo_geocodes, 
                                                     residuals='smoothed_resids', 
                                                     axes='ideologyd4', 
                                                     )

df_wo_geocodes_edu_1 = compare_residuals_by_axes(df=df_wo_geocodes, 
                                                     residuals='smoothed_resids', 
                                                     axes='edu1', 
                                                     )

top_20_resids_group = df_wo_geocodes[df_wo_geocodes['lr_resids'] >= df_wo_geocodes['mlp_resids']
               ].sort_values(by='smoothed_resids', ascending=False).reset_index(drop=False).loc[0:5000]


poor_lr_performance_group = df_wo_geocodes[df_wo_geocodes['lr_resids'] >= df_wo_geocodes['mlp_resids']].sort_values(by='smoothed_resids', ascending=False).reset_index(drop=True).loc[0:500]
poor_lr_performance_group.describe().to_csv('res/poor_lr_performance.csv')



lr_overpred_group = df_wo_geocodes[
    (df_wo_geocodes['lr_resids'] >= df_wo_geocodes['mlp_resids']) & 
    (df_wo_geocodes['y_pred_lr'] > df_wo_geocodes['y_actual'])
    ].sort_values(by=['smoothed_resids', 'y_pred_lr'], ascending=False).reset_index(drop=True).loc[:50].describe().T

lr_overpred_few = df_wo_geocodes[
    (df_wo_geocodes['lr_resids'] >= df_wo_geocodes['mlp_resids']) & 
    (df_wo_geocodes['y_pred_lr'] > df_wo_geocodes['y_actual'])
].sort_values(by=['smoothed_resids', 'y_pred_lr'], ascending=False).reset_index(drop=True).head(n=20)

lr_underpred_group = df_wo_geocodes[
    (df_wo_geocodes['lr_resids'] >= df_wo_geocodes['mlp_resids']) & 
    (df_wo_geocodes['y_pred_lr'] < df_wo_geocodes['y_actual'])
    ].sort_values(by=['smoothed_resids', 'y_pred_lr'], ascending=False).reset_index(drop=True).loc[:50].describe().T

lr_underpred_few = df_wo_geocodes[
    (df_wo_geocodes['lr_resids'] >= df_wo_geocodes['mlp_resids']) & 
    (df_wo_geocodes['y_pred_lr'] < df_wo_geocodes['y_actual'])
].sort_values(by=['smoothed_resids', 'y_pred_lr'], ascending=False).reset_index(drop=True).head(n=20)



show_age_mismatch = pd.concat((lr_underpred_few, lr_overpred_few), ignore_index=True)
show_age_mismatch['age_groups'] = np.argmax(show_age_mismatch[["aged1", "aged2", "aged3", "aged5", "aged6"]], axis=1).reshape(-1, 1)
show_age_mismatch.drop(columns=["aged1", "aged2", "aged3", "aged5", "aged6"], inplace=True)


plt_age_mismatch = show_age_mismatch[['age_groups', 'y_pred_mlp', 'y_pred_lr', 'y_actual']].groupby(by='age_groups').mean()
direction = np.where(plt_age_mismatch['y_pred_lr'] > plt_age_mismatch['y_actual'], 1, -1)
plt_age_mismatch['mlp_lr_pred_diff']= plt_age_mismatch['y_pred_mlp'] - plt_age_mismatch['y_pred_lr']
plt_age_mismatch['unexplained_varr'] = plt_age_mismatch['y_actual'] - plt_age_mismatch['y_pred_lr']
plt_age_mismatch.index = ['age18-24', 'age25-34', 'age35-44', 'age55-64', 'age64+']
plt.bar(x=plt_age_mismatch.index, height=plt_age_mismatch['mlp_lr_pred_diff'], color='navy')
plt.suptitle("Average difference between ML and baseline model predictions for top 100 residuals", color='black')
plt.title("Baseline model under-predicts for younger people and over-predicts for older people", color='red')
plt.axhline(y=0, color='red')
plt.ylabel('Average difference between ML and baseline model predictions')

plt.show()


print(lr_overpred_few.T)
"""Groups that baseline psychological model tends to overpredict in awareness of climate change:
From top 5 observations 
- Typically older people (aged4, aged5)
- People who are neither Black, White, or Hispanic (raced4)
- Generally tend to describe as conservative 
- Q: typically answer all 0s for political affiliation question; does that mean they do not answer or does that mean they are in the middle?
"""

poor_performance_group = df_wo_geocodes[df_wo_geocodes['lr_resids'] <= df_wo_geocodes['mlp_resids']].sort_values(by='smoothed_resids', ascending=False).reset_index(drop=True).loc[0:5000]
poor_performance_group.describe().to_csv('res/poor_mlp_performances.csv')
top_20_resids_group.drop(columns='index', inplace=True)
    
top_20_resids_group.to_csv('res/top_20_residual_groups.csv')
age_cols=["aged1", "aged2", "aged3", "aged5", "aged6"]
top_100_smoothed_resids = df_wo_geocodes.sort_values(by='smoothed_resids', ascending=False).reset_index(drop=True).loc[:100]
top_100_smoothed_resids['y_lr_diff'] = top_100_smoothed_resids['y_actual'] - top_100_smoothed_resids['y_pred_lr']
top_100_smoothed_resids['y_mlp_diff'] = top_100_smoothed_resids['y_actual'] - top_100_smoothed_resids['y_pred_mlp']
top_100_smoothed_resids['age_groups'] = np.argmax(top_100_smoothed_resids[age_cols], axis=1).reshape(-1, 1)
top_100_smoothed_resids.drop(columns=age_cols, inplace=True)

t100_sr_by_age = top_100_smoothed_resids[['y_lr_diff', 'y_mlp_diff', 'age_groups']].groupby(by='age_groups').mean()
t100_sr_by_age.index = ['age18-24', 'age25-34', 'age35-44', 'age55-64', 'age64+']
plt.bar(x=t100_sr_by_age.index, height=t100_sr_by_age['y_lr_diff'], color='lightgreen', edgecolor='black', label='MAE of baseline for\ntop 100 smoothed residuals')
plt.bar(x=t100_sr_by_age.index, height=t100_sr_by_age['y_mlp_diff'], color='darkgreen', edgecolor='black', label='MAE of MLP for\ntop 100 smoothed residuals')
plt.suptitle("Average mean absolute error for ML and baseline model stratified by age", color='black')
plt.title("Baseline model under-predicts for younger people and over-predicts for older people", color='black')
plt.axhline(y=0, color='red')
plt.ylabel('Average difference between ML and baseline model predictions')
plt.legend()
plt.show()

all_zero_mask = (top_100_smoothed_resids[['raced2', 'raced3', 'raced4']] == 0).all(axis=1)
top_100_smoothed_resids['raced1'] = np.where(all_zero_mask, 1, 0)
top_100_smoothed_resids['race'] = np.argmax(top_100_smoothed_resids[['raced2', 'raced3', 'raced1', 'raced4']], axis=1)
t100_sr_by_race = top_100_smoothed_resids[['y_lr_diff', 'y_mlp_diff', 'race']].groupby(by='race').mean()
t100_sr_by_race.index = ['Black', 'Hispanic', 'White', 'Not Black, Hispanic, or White']

plt.bar(x=t100_sr_by_race.index, height=t100_sr_by_race['y_lr_diff'], color='lightgreen', edgecolor='black', label='MAE of baseline for\ntop 100 smoothed residuals')
plt.bar(x=t100_sr_by_race.index, height=t100_sr_by_race['y_mlp_diff'], color='darkgreen', edgecolor='black', label='MAE of MLP for\ntop 100 smoothed residuals')
plt.suptitle("Average mean absolute error for ML and baseline model stratified by race", color='black')
plt.title("Baseline model under-predicts for younger people and over-predicts for older people", color='black')
plt.axhline(y=0, color='red')
plt.ylabel('Average difference between ML and baseline model predictions')
plt.legend()
plt.show()

df_wo_codes.drop(['CC12'], axis=1).columns
#!0, #11, #12, #13



top_100_smoothed_resids['party_identification'] = np.argmax(top_100_smoothed_resids[['PDd1', 'PDd2', 'PDd3', 'PDd5', 'PDd6', 'PDd7']], axis=1)
top_100_smoothed_resids[['party_identification', 'y_pred_mlp', 'y_pred_lr', 'y_actual']].groupby(by='party_identification').mean()

## NOTE: MLP model is performing better with age.. Why is that? 

trial_vec = np.zeros_like(df_wo_codes.drop(['CC12'], axis=1).columns)

rp = np.zeros_like(X[0])
len(df_wo_codes.drop(columns=['CC12'], axis=1).columns)
len(rp)
rp
with torch.no_grad(): 
    out = model(torch.from_numpy(rp).float())
    
rp[10] = 1
with torch.no_grad():
    inj = model(torch.from_numpy(rp).float())
inj-out
    

def pass_each_input(trial_vec, positions_to_probe, model):
    df = pd.DataFrame(index=positions_to_probe, columns=['estimated_coeffs'])

    # Make sure trial_vec is a copy so we don't overwrite the original
    base_vec = trial_vec.copy()

    for pos in positions_to_probe:
        # Set to 0
        input_vec_0 = base_vec.copy()
        input_vec_0.reshape(-1, 1)[pos] = 0
        
        # Set to 1
        input_vec_1 = base_vec.copy()
        input_vec_1.reshape(-1, 1)[pos] = 1

        with torch.no_grad():
            null_out = model(torch.from_numpy(input_vec_0).float().unsqueeze(0))
            probed_out = model(torch.from_numpy(input_vec_1).float().unsqueeze(0))
        
        # Compute change
        est_coefficient = (probed_out - null_out).item()
        df.loc[pos, 'estimated_coeffs'] = est_coefficient

    return df



df_est_coeffs = pass_each_input(trial_vec=np.zeros((1, 32)),
                            positions_to_probe=np.arange(0, 32, 1), 
                            model=model)

trial_vec_male =np.zeros((1, 32))
trial_vec_male.reshape(-1, 1)[1] = 1 # Black = 1 

output = pass_each_input(trial_vec, positions_to_probe=np.arange(0, 32, 1), model=model).values
df_est_coeffs.drop(columns=['effects_for_males'], axis=1, inplace=True)

trial_vec_black = np.zeros((1, 32))
trial_vec_black.reshape(-1, 1)[6] = 1 # black
black_x_other_effects = pass_each_input(trial_vec_black, positions_to_probe=np.arange(0, 32, 1), model=model).values

df_est_coeffs['black_x_coeffs'] = black_x_other_effects
df_est_coeffs
df_est_coeffs['lr_coeffs'] = lr.coef_.reshape(-1, 1)
df_est_coeffs

df_wo_codes.columns
df_est_coeffs.index = ['max_temp', 'gender', 'warm_strongcool', 'warm_cool', 'cool_warm', 'cool_strongwarm', 'black', 'hispan', 'other_race', 'v_lib', 'lib', 'cons', 'v_cons', 
                       'strong_dem', 'dem', 'lean_dem', 'lean_rep', 'rep', 'strong_rep', 'HS_or_less', 'college', 'post_grad', 
                       'no_church', 'seldom_church', 'few_church', 'week_church', 'week_church_s', '18-24', '25-34', '35-44', '55-64', '64+']
plt.plot(df_est_coeffs.index, df_est_coeffs['estimated_coeffs'], color='blue', label='Probed MLP coefficients')
plt.plot(df_est_coeffs.index, df_est_coeffs['lr_coeffs'], color='red', label='Baseline model coefficients')
plt.title('Effects sizes for 32 coefficients/explanatory variables\n ML model yields more precise estimates')
plt.xticks(rotation=60, ha='right')
plt.xlabel("Explanatory variables")
plt.ylabel("Beta coefficients")
plt.tight_layout()
plt.legend()
plt.show()     

plt.plot(df_est_coeffs.index, df_est_coeffs['estimated_coeffs'], color='blue', label='Probed MLP coefficients')
plt.plot(df_est_coeffs.index, df_est_coeffs['lr_coeffs'], color='red', label='Baseline model coefficients')
plt.plot(df_est_coeffs.index, black_x_other_effects, color='purple', label='Raced = 1')
plt.title('Effects sizes for 32 coefficients/explanatory variables\n ML model yields more precise estimates')
plt.xticks(rotation=60, ha='right')
plt.xlabel("Explanatory variables")
plt.ylabel("Beta coefficients")
plt.tight_layout()
plt.legend()
plt.show()    


trial_vec_hispan = np.zeros((1, 32))
trial_vec_hispan.reshape(-1, 1)[7] = 1 # black
hispan_x_other_effects = pass_each_input(trial_vec_hispan, positions_to_probe=np.arange(0, 32, 1), model=model).values


plt.plot(df_est_coeffs.index, df_est_coeffs['estimated_coeffs'], color='blue', label='Probed MLP coefficients')
plt.plot(df_est_coeffs.index, df_est_coeffs['standard'], color='red', label='Baseline model coefficients')
plt.plot(df_est_coeffs.index, hispan_x_other_effects, color='purple', label='Raced3 = 1')
plt.title('Effects sizes for 32 coefficients/explanatory variables\n ML model yields more precise estimates')
plt.xticks(rotation=60, ha='right')
plt.xlabel("Explanatory variables")
plt.ylabel("Beta coefficients")
plt.tight_layout()
plt.legend()
plt.show()    
# Critique linear model: ideology and race 



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




standardized_est_coeffs = fit_new_linear_reg_model(df_wo_codes) 
df_est_coeffs
create_interaction_terms(df_wo_codes)

critique_df_1 = df_wo_codes.copy()
ideology_cont = np.argmax(df_wo_codes[['ideologyd1', 'ideologyd2', 'ideologyd4']], axis=1)
critique_df_1['ideocont_x_black'] = ideology_cont * df_wo_codes['raced2']
critique_df_1['ideocont_x_hispan'] = ideology_cont * df_wo_codes['raced3']
critique_df_1['ideocont_x_neither'] = ideology_cont * df_wo_codes['raced4']

coefs_for_continuous_ideoxrace = fit_new_linear_reg_model(critique_df_1)

# cat1 has the raced2,raced3, raced4

df_with_race_ideology_vars = create_interaction_terms(df_wo_codes, 
                                                     cat1=['raced2', 'raced3', 'raced4'], 
                                                     cat2=['ideologyd1', 'ideologyd2', 'ideologyd4', 'ideologyd5'], 
                                                     prefix='int')


    
coefs_mod_race_ideology_int = fit_new_linear_reg_model(df_with_race_ideology_vars)
df_est_coeffs['race_ideology_interaction'] = coefs_mod_race_ideology_int[:32]

df_race_ideo_gender_int = create_interaction_terms(df_with_race_ideology_vars, cat1=['Gender'], cat2=['raced2', 'raced3', 'raced4'], prefix="int")

coefs_mod_race_ideology_gender_int = fit_new_linear_reg_model(df_race_ideo_gender_int)
df_est_coeffs['rxideo_rxgender']= coefs_mod_race_ideology_gender_int[:32]

df_race_ideo_gender_edu_aged_int = create_interaction_terms(df_race_ideo_gender_int, cat1=['aged1', 'aged2', 'aged3', 'aged5', 'aged6'],
                             cat2=['edu1', 'edu3', 'edu4'], prefix='int')

coefs_mod_r_i_g_edu_aged = fit_new_linear_reg_model(df_race_ideo_gender_edu_aged_int)

df_est_coeffs['rxideo_rxgen_edxage'] = coefs_mod_r_i_g_edu_aged[:32]

recursive_df3= create_interaction_terms(df_race_ideo_gender_edu_aged_int, 
                         cat1=['raced2', 'raced3', 'raced4'], 
                         cat2=['attendd1', 'attendd2', 'attendd3', 'attendd5', 'attendd6'], prefix='int')

coefs_rec_3 = fit_new_linear_reg_model(recursive_df3)

df_est_coeffs['rx_ideo_rxgen_edxage_rxchurch'] = coefs_rec_3[:32]


recursive_df4 = create_interaction_terms(recursive_df3, cat1=['aged1', 'aged2', 'aged3', 'aged5', 'aged6'],
                         cat2=['ideologyd1', 'ideologyd2', 'ideologyd4', 'ideologyd5'], prefix='int')
coefs_rec_4 = fit_new_linear_reg_model(recursive_df4)

plt.plot(df_est_coeffs.index, df_est_coeffs['estimated_coeffs'], color='blue', label='Probed MLP coefficients')
plt.plot(df_est_coeffs.index, standardized_est_coeffs, color='red', label='Baseline model coefficients')
plt.plot(df_est_coeffs.index, df_est_coeffs['race_ideology_interaction'], color='lightsalmon', label='Baseline + Race x Ideology (12) Interactions')
plt.plot(df_est_coeffs.index, df_est_coeffs['rxideo_rxgender'], color='purple', label='... ')
plt.plot(df_est_coeffs.index, df_est_coeffs['rxideo_rxgen_edxage'], color='black')
plt.plot(df_est_coeffs.index, df_est_coeffs['rx_ideo_rxgen_edxage_rxchurch'], color='green')
plt.plot(df_est_coeffs.index, coefs_rec_4[:32], color='purple',label='Interaction btwn age and race')
plt.title('Effects sizes for 32 coefficients/explanatory variables\n ML model yields more precise estimates')
plt.xticks(rotation=60, ha='right')
plt.xlabel("Explanatory variables")
plt.ylabel("Beta coefficients")
plt.tight_layout()
plt.legend()
plt.show()     

df_with_agexideology = create_interaction_terms(df_wo_codes, cat1=['aged1', 'aged2', 'aged3', 'aged5', 'aged6'],
                         cat2=['ideologyd1', 'ideologyd2', 'ideologyd4', 'ideologyd5'], prefix='int')

age_ideo_coefs, age_ideo_preds = fit_new_linear_reg_model(df_with_agexideology)

df_wo_geocodes['y_lr_agexideo_pred'] = age_ideo_preds
df_wo_geocodes[['y_lr_agexideo_pred', 'y_pred_lr', 'y_pred_mlp']]
df_wo_geocodes['y_agexideo_diff'] = df_wo_geocodes['y_actual'] - df_wo_geocodes['y_lr_agexideo_pred']

top_100_smoothed_resids = df_wo_geocodes.sort_values(by='smoothed_resids', ascending=False).reset_index(drop=True).loc[:100]
top_100_smoothed_resids['y_lr_diff'] = top_100_smoothed_resids['y_actual'] - top_100_smoothed_resids['y_pred_lr']
top_100_smoothed_resids['y_mlp_diff'] = top_100_smoothed_resids['y_actual'] - top_100_smoothed_resids['y_pred_mlp']
top_100_smoothed_resids['age_groups'] = np.argmax(top_100_smoothed_resids[age_cols], axis=1).reshape(-1, 1)
top_100_smoothed_resids.drop(columns=age_cols, inplace=True)

t100_sr_by_age = top_100_smoothed_resids[['y_lr_diff', 'y_mlp_diff', 'y_agexideo_diff','age_groups']].groupby(by='age_groups').mean()
t100_sr_by_age.index = ['age18-24', 'age25-34', 'age35-44', 'age55-64', 'age64+']
plt.bar(x=t100_sr_by_age.index, height=t100_sr_by_age['y_lr_diff'], color='lightgreen', edgecolor='black', label='MAE of baseline for\ntop 100 smoothed residuals')
plt.bar(x=t100_sr_by_age.index, height=t100_sr_by_age['y_agexideo_diff'], color='mediumseagreen', edgecolor='black', label='MAE of baseline + SRM critique')
plt.bar(x=t100_sr_by_age.index, height=t100_sr_by_age['y_mlp_diff'], color='darkgreen', edgecolor='black', label='MAE of MLP for\ntop 100 smoothed residuals')

plt.suptitle("Average mean absolute error for ML and baseline model stratified by age", color='black')
plt.title("Baseline model under-predicts for younger people and over-predicts for older people", color='black')
plt.axhline(y=0, color='red')
plt.ylabel('Average difference between ML and baseline model predictions')
plt.legend()
plt.show()

all_zero_mask = (top_100_smoothed_resids[['raced2', 'raced3', 'raced4']] == 0).all(axis=1)
top_100_smoothed_resids['raced1'] = np.where(all_zero_mask, 1, 0)
top_100_smoothed_resids['race'] = np.argmax(top_100_smoothed_resids[['raced2', 'raced3', 'raced1', 'raced4']], axis=1)
t100_sr_by_race = top_100_smoothed_resids[['y_lr_diff', 'y_mlp_diff', 'y_agexideo_diff', 'race']].groupby(by='race').mean()
t100_sr_by_race.index = ['Black', 'Hispanic', 'White', 'Not Black, Hispanic, or White']

plt.bar(x=t100_sr_by_race.index, height=t100_sr_by_race['y_lr_diff'], color='lightgreen', edgecolor='black', label='MAE of baseline for\ntop 100 smoothed residuals')
plt.bar(x=t100_sr_by_race.index, height=t100_sr_by_race['y_agexideo_diff'], color='mediumgreen', edgecolor='black', label="SRM intervention")
plt.bar(x=t100_sr_by_race.index, height=t100_sr_by_race['y_mlp_diff'], color='darkgreen', edgecolor='black', label='MAE of MLP for\ntop 100 smoothed residuals')
plt.suptitle("Average mean absolute error for ML and baseline model stratified by race", color='black')
plt.title("Baseline model under-predicts for younger people and over-predicts for older people", color='black')
plt.axhline(y=0, color='red')
plt.ylabel('Average difference between ML and baseline model predictions')
plt.legend()
plt.show()

# 1.7892 

df_est_coeffs
## Interaction term between IDEOLOGY AND EDUCATION 

## Interaction term between IDEOLOGY and CHURCH GOING STATUS (multiplicative effect) 


# ['Gender','ideologyd1', 'ideologyd2', 'ideologyd4', 'ideologyd5', 'edu1', 'edu3', 'raced2', 
            #                          'raced3', 'raced4', 'attendd1', 'attendd2', 'attendd3', 'attendd5', 'attendd6',
           ##                           'aged1', 'aged2', 'aged3', 'aged5', 'aged6',
             #                         'PDd1', 'PDd2', 'PDd3', 'PDd5', 'PDd6', 'PDd7']

exclude_cols = ['smoothed_resids', 'mlp_resids', 'lr_resids', 'y_pred_lr', 'y_pred_mlp', 'y_actual']


df_k_greatest_resids = lookup_features_from_top_k_residuals(top_20_resids_group, 
                                     [col for col in df_wo_geocodes.columns if col not in exclude_cols],
                                     'smoothed_resids', 
                                      k=5000)



df_k_greatest_resids.to_csv('res/smoothed_residual_distributions.csv')

df_k_greatest_resids['smoothed_resids'].loc[0]
df_wo_geocodes['PDd1'].value_counts()





"""Restore and/or retrieve training"""
## Restore a past training session.
exp_path = "/Users/stanleyhuang/Desktop/01 Projects/YAB/cds_climate_change_perception/Climate Change Perceptions/checkpoints/env_pred_tpe_algo_run"
results= tune.Tuner.restore(exp_path, trainable=new_train_fn).get_results()

## Get the best result by looking at the metrics you reported 
best_result = results.get_best_result(metric="valid_loss", mode="min")
checkpoint: Checkpoint = best_result.checkpoint

# Access directory path (if you saved files manually in training)
checkpoint_path = checkpoint.to_directory()

# Example: load training history or model state from that path 
best_configs = results.get_best_result(metric='valid_loss', mode='min').config
merged_configs = {**train_configs, **best_configs}
model_data = torch.load(os.path.join(checkpoint_path, "model.pth"))
model = initialize_model(merged_configs)  
model.load_state_dict(model_data)


"""Plot validation loss as a function of training size."""
data = pd.read_csv(data_file_path_geocoded)
shuffled_data = data.sample(frac=1).reset_index(drop=True) 
cols_to_keep = [col for col in shuffled_data.columns if not col.startswith('cd')]
shuffled_data= shuffled_data[cols_to_keep]
X_c = shuffled_data.drop(['CC12'], axis=1).values.astype(np.float32)
y_c = shuffled_data['CC12'].values.reshape(-1, 1).astype(np.float32)
    
plot_linear_training_validation_loss(X_c, y_c, intervals=10)
best_model= initialize_model(merged_configs)
plot_nn_training_validation_loss(X_c, y_c, intervals=10, configs=merged_configs)


"""Plot residuals of neural network model and baseline psychological model"""

## Reload the best model
best_r2_model = results.get_best_result('best_r2_score', mode='max')
best_configs = best_r2_model.config
merged_configs = {**best_configs, **train_configs}
cp_path = best_r2_model.checkpoint.path
model_data = torch.load(os.path.join(cp_path, 'model.pth'))
model = initialize_model(merged_configs)
model.load_state_dict(model_data)

model.state_dict()

df = pd.read_csv(train_configs['file_path'])
sc = StandardScaler()
X = df.drop(['CC12'], axis=1).values
y = df['CC12'].values - 1
X = sc.fit_transform(X)

lr = LinearRegression()
lr.fit(X, y)

y_pred_lr = lr.predict(X)
lr.score(X, y)

model.eval()
with torch.no_grad():
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    y_pred_mlp = model(X_t)
    r2_scorer = R2Score()
    r2_scorer.update(y_pred_mlp, y_t.reshape(-1, 1))
    r2_score_eval = r2_scorer.compute()

r2_score_eval
    

"""Constructing a dataset with actual y value, baseline y, and MLP-computed y"""
  
y = y.reshape(-1, 1)
y_pred_lr = y_pred_lr.reshape(-1, 1)
y_pred_mlp_np = y_pred_mlp.cpu().numpy() 

preds_data = np.hstack((y, y_pred_lr, y_pred_mlp_np))

compare_resids = pd.DataFrame(data=preds_data, columns=['y_actual', 'y_pred_lr', 'y_pred_mlp'])

## compute true residuals 
compare_resids['lr_resids'] = compute_true_residuals(compare_resids['y_actual'], compare_resids['y_pred_lr'])
compare_resids['mlp_resids'] = compute_true_residuals(compare_resids['y_actual'], compare_resids['y_pred_mlp'])

## compute smoothed residuals 

compare_resids['smoothed_resids'] = compute_smoothed_residuals(y=compare_resids['y_actual'],
                                                                baseline_y=compare_resids['y_pred_lr'],
                                                                alg_y=compare_resids['y_pred_mlp'])




y_actual = compare_resids['y_actual'] 
lin_mlp_diff = compare_resids['y_pred_lr'] - compare_resids['y_pred_mlp']
plt.hist(lin_mlp_diff)
plt.title('Absolute Linear-MLP difference')
plt.show()

sorted(y_actual.unique())
figs, ax = plt.subplots(nrows=5, figsize=(8, 10))
ax = ax.flatten()

for i, cat in enumerate(sorted(compare_resids['y_actual'].unique())):
    # Filter rows where y_actual == cat
    subset = compare_resids[compare_resids['y_actual'] == cat]

    # Extract predictions
    cat_lin_pred = subset['y_pred_lr']
    cat_mlp_pred = subset['y_pred_mlp']

    # Compute differences
    lin_mlp_diff_for_cat = cat_lin_pred - cat_mlp_pred

    # Plot
    ax[i].hist(lin_mlp_diff_for_cat, color='skyblue', edgecolor='black')
    ax[i].set_title(f"Category {cat}")
    ax[i].axvline(0, color='red', linestyle='--')

plt.tight_layout()
plt.show()




sorted(compare_resids['smoothed_resids'])[-3:-3]

df_geo = pd.read_csv(data_file_path_geocoded)

combined_df = pd.concat([df_geo, compare_resids], axis=1)



compare_residuals_by_axes(combined_df, residuals='smoothed_resids', axes='Gender', 
                          mapping={0.0: 'm', 1.0: 'f'})

compare_residuals_by_axes(combined_df, residuals='smoothed_resids', axes='')

combined_df[combined_df['Gender'] == 0.0]['smoothed_resids'].describe()['count']

