import numpy as np 
import pandas as pd 
import torch
from torch import nn
import os
import torch.optim as optim
from sklearn.metrics import r2_score
from skorch import NeuralNetRegressor
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, Callback, EpochScoring
import ray
import tempfile
import time
import ray
from ray.tune import Checkpoint
import matplotlib.pyplot as plt


def reset_model_parameters(model):
    for module in model.modules(): 
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters() 
  
class DynamicNet(nn.Module): 
    def __init__(self, input_dims, list_n_units, output_dims, activation_fn=nn.ReLU): 
        super().__init__()
        self.list_n_units = list_n_units

        layers = []        
        layers.append(nn.Linear(input_dims, list_n_units[0]))
        layers.append(activation_fn())
        
        for i in range(len(list_n_units) - 1): 
            layers.append(nn.Linear(list_n_units[i], list_n_units[i+1]))
            layers.append(activation_fn())
        
        layers.append(nn.Linear(list_n_units[-1], output_dims))
        
        self.neural_network = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.neural_network(x)
    
    def __str__(self): 
        str_val = "".join(["_"+str(val) for val in self.list_n_units])
        return f"dnn_{len(self.list_n_units)}{str_val}"



class TopKSparsify(Callback): 
    def __init__(self, k, layers_to_sparsify=1, epochs_to_sparsify=5): 
        super().__init__()
        self.k = k
        self.layers_to_sparsify = layers_to_sparsify
        self.epochs_to_sparsify = epochs_to_sparsify
    
    def apply_top_k_sparsification(self, x, k): 
        x_flat_abs = x.flatten().abs()
        
        _, top_k_indices = torch.topk(x_flat_abs, k=k)
    
        mask = torch.zeros_like(x_flat_abs)
        mask[top_k_indices] = 1
        
        mask = mask.view_as(x)
        return mask * x 
    
    def on_epoch_end(self, skorch_model, **kwargs): ## Question should we pass skorch_model in the beginning as an object? 
        current_epoch = skorch_model.history[-1]['epoch']
        if current_epoch % self.epochs_to_sparsify != 0: 
            return 
        
        count = 0
        for name, params in skorch_model.module_.named_parameters():
            if count >= self.layers_to_sparsify:
                break
            if 'weight' in name:
                print(f"Applying top-k sparsification at epoch {current_epoch}...")
                start = time.time()
                masked_res = self.apply_top_k_sparsification(params, self.k)
                with torch.no_grad():
                    params.data.copy_(masked_res)
                end = time.time()
                print(f"Sparsification took {end - start:.4f} seconds")
                count += 1

                

def load_dataset(filepath): 
    return pd.read_csv(
       filepath
        )
    
def initialize_model(configs): 
        pretrained_model = configs.get('pretrain', None)
        val = configs.get('activation_fn', 0) # 0 will represent nn.ReLU
        activation_dct = {
            0: nn.ReLU,
            1: nn.Tanh,
        }
        base_neurons = 2 ** configs['base_exp']     # 8, 16, 32, ..., 4096
        list_n_units = [base_neurons] * configs['num_layers']

        net = DynamicNet(input_dims=configs['input_dims'],
                   list_n_units=list_n_units,
                   activation_fn=activation_dct.get(val, nn.ReLU),
                   output_dims=1)
        if pretrained_model: 
            model_data = torch.load(pretrained_model)
            net.load_state_dict(model_data)
        
        return net 


### Deprecated 
## NOTE: Decided not to use Skorch wrapper 
def controlled_binary_sample(df, binary_conditions, sample_size, seed=17):
    """
    binary_conditions: dict of {column_name: desired proportion of 1s}
    """
    df_sample = pd.DataFrame()
    
    for col, proportion in binary_conditions.items():
        n_1 = int(sample_size * proportion)
        n_0 = sample_size - n_1

        df_1 = df[df[col] == 1].sample(n=n_1,random_state=seed)
        df_0 = df[df[col] == 0].sample(n=n_0, random_state=seed)

        df_col_sample = pd.concat([df_1, df_0])
        df_sample[col] = df_col_sample[col].values  # Just keep the column for now

    return df.loc[df_sample.index].reset_index(drop=True)

def train_network(config): 
    # Load the dataset 
    merged_config = {**config}
    ds = load_dataset(merged_config['file_path'])
    ## Right now there is no distributed training because the data is not being batched in a way that
    ## is accessible by Ray workers... 
    # Prepare dataset 
    X = ds.drop(['CC12'], axis=1).values 
    y = ds['CC12'].values.reshape(-1, 1) # reshape into a 1-d numpy array 
    
    X_np = X.astype(np.float32) # Float 
    y_np = y.astype(np.float32) # Float 
    
    # Initialize neural network model instance 
    nn_model = create_skorch_wrapper(merged_config)
    nn_model.fit(X_np, y_np)
    
    best_epoch_history = min(nn_model.history, key=lambda x: x['valid_loss'])
    valid_loss = best_epoch_history['valid_loss']
    
    ray.tune.report({"valid_loss": valid_loss})
    
    ## Session reporting done in Callback (see on_epoch_end)
    ## Final end of model reporting one in Callback (see on_train_end)


def plot_training_size_loss(training_size, loss_values):    
    plt.figure(figsize=(10, 6))
    plt.plot(training_size, loss_values, marker='o')
    plt.line()
    plt.title('Training Size vs Loss')
    plt.xlabel('Training Size')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
  
        
def create_skorch_wrapper(config):
        
    if config.get('use_callbacks', True):
        callbacks = [
            ('r2_tracker', EpochScoring(scoring=r2_score, 
                                        lower_is_better=False,
                                        name='r2',
                                        use_caching=True,
                                        )),
            ('early_stopping', EarlyStopping(monitor='valid_loss', 
                                             patience=7)),
            # ('save_best_params', Checkpoint( # We don't need checkpoint here because the reporter does it
            #     monitor='valid_loss_best',
            #     f_params='best_weights.pt',
            #     dirname=os.path.join(os.getcwd(), 'checkpoints'))),
            ('ray_reporter', RayReportCallback()),
        ]

        if config.get('use_topk', False):
            if 'k' not in config:
                print('Other options include `layers_to_sparsify`, `epochs_to_sparsify`')
                raise ValueError("Make sure to specify number of K values.")
            
            callbacks.append((
                'top_k_sparsify',
                TopKSparsify(
                    epochs_to_sparsify=config.get('epochs_to_sparsify', 5),
                    k=config['k'],
                    layers_to_sparsify=config.get('layers_to_sparsify', 1)
                )
            ))
    else: 
        callbacks = []
        
    base_neurons = 2 ** config['base_exp']     # 8, 16, 32, ..., 4096
    config['list_n_units'] = [base_neurons] * config['num_layers']
        
    return NeuralNetRegressor(
        module=DynamicNet,
        criterion=nn.MSELoss,
        module__input_dims=config['input_dims'],
        module__list_n_units=config['list_n_units'],
        module__output_dims=1,
        max_epochs=config['max_epochs'],
        optimizer=optim.Adam,
        optimizer__lr=config['lr'],
        callbacks=callbacks,
        train_split=ValidSplit(0.2),
        verbose=1, 
        batch_size=config.get('batch_size', 128))
    
                
class RayReportCallback(Callback): 
    def __init__(self, reporter=ray.tune.report): 
        super().__init__()
        self.reporter = reporter
    
    def on_epoch_end(self, model, **kwargs): 
        last_epoch = model.history[-1]
        
        
        validation_loss = last_epoch.get('valid_loss')
        train_loss = last_epoch.get('train_loss')
        r_squared = last_epoch.get('valid_r2')
        
        report_dict = {
            'train_loss': train_loss,
            'valid_loss': validation_loss,
            'valid_r2': r_squared,
            'step': last_epoch['epoch']
        }
        
        report_dict = {k: v for k, v in report_dict.items() if v is not None}
        if get_context().get_world_rank() == 0: #Only one worker reports training
            
            with tempfile.TemporaryDirectory() as temp_dir: 
                file_path = os.path.join(temp_dir, "model.pt")
                torch.save(model.module_.state_dict(), file_path)
            if self.reporter: 
                self.reporter(report_dict, 
                              checkpoint=Checkpoint.from_directory(temp_dir))
            else: 
                print('Valdiation loss:', validation_loss)
                