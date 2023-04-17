import configparser
import numpy as np

import data.generation as gen

'''
generate.py

The purpose of this file is solely to generate data for our model.
Please make sure that this file is run before train.py.
The data is stored in a file specific to the combination of:
    {num_hours, num_days, num_weeks}
    i.e. {1, 2, 3} => 1_2_3.npz

You may have to modify get_filepath() in data/generation.py,
if you want to test with different datasets,
as this naming system does not account for it.
'''

if __name__ == '__main__':
    #===============
    # Config parsing
    #===============

    config = configparser.ConfigParser()
    config.read('model.config')
    config = config['Default']
    
    adj_filename = config['adj_filename']
    traffic_filename = config['traffic_filename']
    weather_filename = config['weather_filename']
    
    print(f'Adjacency matrix filename: {adj_filename}.')
    print(f'Traffic data filename: {traffic_filename}.')
    print(f'Weather data filename: {weather_filename}')
    print('')
    
    num_hours = int(config['num_hours'])
    num_days = int(config['num_days'])
    num_weeks = int(config['num_weeks'])
    pred_window_size = int(config['pred_window_size'])
    
    #================
    # Retrieving data
    #================
    
    print(f'Predicting using information from {num_hours} hours, {num_days} days, and {num_weeks} weeks prior to prediction period.')
    
    savefile = gen.get_filepath(num_hours, num_days, num_weeks)

    print(f'Data will be saved to {savefile}')
    print('')
    
    # Retrieve generic pool of traffic data
    X_h, X_d, X_w, y = gen.generate_traffic_data(traffic_filename, num_hours, num_days, num_weeks, pred_window_size)
    
    print('Successfully generated traffic dataset')
    
    # Retrieve weather data
    W = gen.generate_weather_data(weather_filename, num_hours, num_days, num_weeks, pred_window_size)
    
    print('Successfully generated weather dataset')
    
    # Retrieve adjacency matrix
    A = gen.generate_adjacency_matrix(adj_filename, traffic_filename)
    
    print('Successfully generated adjacency matrix')
    
    #============
    # Saving data
    #============
        
    np.savez_compressed(savefile, hourly = X_h, daily = X_d, weekly = X_w, pred = y, weather = W, adj_mx = A)
    