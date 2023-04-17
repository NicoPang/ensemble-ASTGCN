import configparser
import gc
import numpy as np
import time

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

import data.generation as gen
import model.ensembleASTGCN as models

def train_one_epoch(model, training_loader):
    return 0

def train_test(model, training_loader, testing_loader, num_epochs):
    
    for epoch in range(num_epochs):
        print(f'BEGIN: Epoch {epoch + 1}')
        
#        model.train(True)
        
        loss = train_one_epoch(model, training_loader)
        
#        model.train(False)
        

if __name__ == '__main__':
    #===============
    # Config parsing
    #===============

    config = configparser.ConfigParser()
    config.read('model.config')
    config = config['Default']
    
    num_hours = int(config['num_hours'])
    num_days = int(config['num_days'])
    num_weeks = int(config['num_weeks'])
    pred_window_size = int(config['pred_window_size'])
    
    K = int(config['K'])
    
    epochs = int(config['epochs'])
    
    #===============
    # Data retreival
    #===============
    
    savefile = gen.get_filepath(num_hours, num_days, num_weeks)
    
    print(f'Retreiving data from {savefile}')
    
    packaged_data = np.load(savefile)
    
#    X_h = torch.from_numpy(packaged_data['hourly'])
#    X_d = torch.from_numpy(packaged_data['daily'])
#    X_w = torch.from_numpy(packaged_data['weekly'])
#    W = torch.from_numpy(packaged_data['weather'])
#    y = torch.from_numpy(packaged_data['pred'])
#
#    A = torch.from_numpy(packaged_data['adj_mx'])
    
    X_h = np.copy(packaged_data['hourly'])
    X_d = np.copy(packaged_data['daily'])
    X_w = np.copy(packaged_data['weekly'])
    W = np.copy(packaged_data['weather'])
    y = np.copy(packaged_data['pred'])

    A = np.copy(packaged_data['adj_mx'])

    packaged_data = None

    print(f'Hourly traffic dims: {X_h.shape}')
    print(f'Daily traffic dims: {X_d.shape}')
    print(f'Weekly traffic dims: {X_w.shape}')
    print(f'Weather dims: {W.shape}')
    print(f'Prediction dims: {y.shape}')
    print(f'Adjacency matrix dims: {A.shape}')
    print('')
    
    #===============
    # Data splitting
    #===============
    
    data_indices = list(range(y.shape[0]))

    train_ind, test_ind = train_test_split(data_indices)

    print(f'Training size: {len(train_ind)}')
    print(f'Testing size: {len(test_ind)}')
    
    train_xh = torch.from_numpy(np.copy(X_h[train_ind]))
    test_xh = torch.from_numpy(np.copy(X_h[test_ind]))
    X_h = None
    train_xd = torch.from_numpy(np.copy(X_d[train_ind]))
    test_xd = torch.from_numpy(np.copy(X_d[test_ind]))
    X_d = None
    train_xw = torch.from_numpy(np.copy(X_w[train_ind]))
    test_xw = torch.from_numpy(np.copy(X_w[test_ind]))
    X_w = None
    train_w = torch.from_numpy(np.copy(W[train_ind]))
    test_w = torch.from_numpy(np.copy(W[test_ind]))
    W = None
    train_y = torch.from_numpy(np.copy(y[train_ind]))
    test_y = torch.from_numpy(np.copy(y[test_ind]))
    y = None
    
    
    training_dataset = TensorDataset(train_xh, train_xd, train_xw, train_w, train_y)
    testing_dataset = TensorDataset(test_xh, test_xd, test_xw, test_w, test_y)
    
    training_loader = DataLoader(training_dataset, batch_size = 64, shuffle = True)
    testing_loader = DataLoader(testing_dataset, batch_size = 32)

    print(f'Training X_h shape: {train_xh.shape}')
    print(f'Testing X_h shape: {test_xh.shape}')
    print(f'Training X_d shape: {train_xd.shape}')
    print(f'Testing X_d shape: {test_xd.shape}')
    print(f'Training X_w shape: {train_xw.shape}')
    print(f'Testing X_w shape: {test_xw.shape}')
    print(f'Training W shape: {train_w.shape}')
    print(f'Testing W shape: {test_w.shape}')
    print(f'Training y shape: {train_y.shape}')
    print(f'Testing y shape: {test_y.shape}')
    print('')
    
    train_xh = None
    test_xh = None
    train_xd = None
    test_xd = None
    train_xw = None
    test_xw = None
    train_w = None
    test_w = None
    train_y = None
    test_y = None
    
    print(f'Training dataset size: {len(training_dataset)}')
    print(f'Testing dataset size: {len(testing_dataset)}')
    
    #===============
    # Model Creation
    #===============
    
    # TODO make model
    model = 1
    
    #===================
    # Training + Testing
    #===================
    
    train_test(model, training_loader, testing_loader, epochs)

    print('Sleeping...')
    time.sleep(10)
