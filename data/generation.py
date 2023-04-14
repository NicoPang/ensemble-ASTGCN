import numpy as np

# Global variables
# timestamps_per_hour is equivalent to prediction window size
timestamps_per_week = 2016
timestamps_per_day = 288

# returns filepath for specific parameters
def get_filepath(num_hours, num_days, num_weeks):
    return f'{num_hours}_{num_days}_{num_weeks}.npz'
    
def get_required_diff(hours, days, weeks, pred_window_size):
    '''
    Returns the window size of one data point
    This window will metaphorically slide across the dataset, generating data
    '''
    
    # Typically you would expect the week to be the longest.
    # This is designed to accomodate for faulty input.
    
    week_window_size = timestamps_per_week * weeks
    day_window_size = timestamps_per_day * days
    hour_window_size = pred_window_size * hours
    return max(week_window_size, day_window_size, hour_window_size) + pred_window_size

def generate_traffic_data(file, hours, days, weeks, pred_window_size):
    '''
    Returns four arrays of dims ([S, N, F, T_h], [S, N, F, T_d], [S, N, F, T_w], [S, N, T_p])
        S   -> number of datapoints total
        N   -> number of nodes. For PEMS04, this is 307
        F   -> number of features per node. For PEMS04, this is 3
        T_h -> timestamps for hourly channel
        T_d -> timestamps for daily channel
        T_w -> timestamps for weekly channel
        T_p -> timestamps for prediction output
    '''
    
    data = np.load(file)['data'] # [16992, 307, 3]
    
    # Get range of possible start/stops given the data
    window_size = get_required_diff(hours, days, weeks, pred_window_size)
    num_datapoints = data.shape[0] - window_size + 1
    
    print(f'Window size: {window_size}')
    print(f'Generating {num_datapoints} datapoints')
    
    # output arrays
    X_h = []
    X_d = []
    X_w = []
    y = []
    
    for i in range(num_datapoints):
        t_0 = i + window_size - pred_window_size
        
        temp_xh = []
        temp_xd = []
        temp_xw = []
        temp_y = []
        
        # Hourly
        for hour in range(hours, 0, -1):
            start = t_0 - hour * pred_window_size
            for i in range(pred_window_size):
                temp_xh.append(data[start + i])
        
        # Daily
        for day in range(days, 0, -1):
            start = t_0 - day * timestamps_per_day
            for i in range(pred_window_size):
                temp_xd.append(data[start + i])
            
        # Weekly
        for week in range(weeks, 0, -1):
            start = t_0 - week * timestamps_per_week
            for i in range(pred_window_size):
                temp_xw.append(data[start + i])
                
        # Expected output
        for i in range(pred_window_size):
            temp_y.append(data[t_0 + i][:,1])
            
        # Reshape to fit data
        temp_xh = np.expand_dims(np.array(temp_xh, dtype = np.float32), axis = 0).transpose(0, 2, 3, 1)
        temp_xd = np.expand_dims(np.array(temp_xd, dtype = np.float32), axis = 0).transpose(0, 2, 3, 1)
        temp_xw = np.expand_dims(np.array(temp_xw, dtype = np.float32), axis = 0).transpose(0, 2, 3, 1)
        temp_y = np.expand_dims(np.array(temp_y, dtype = np.float32), axis = 0).transpose(0, 2, 1)
        
        X_h.append(temp_xh)
        X_d.append(temp_xd)
        X_w.append(temp_xw)
        y.append(temp_y)
        
    X_h = np.array(X_h, dtype = np.float32)
    X_d = np.array(X_d, dtype = np.float32)
    X_w = np.array(X_w, dtype = np.float32)
    y = np.array(y, dtype = np.float32)
        
    print(f'X_h size: {X_h.shape}')
    print(f'X_d size: {X_d.shape}')
    print(f'X_w size: {X_w.shape}')
    print(f'y size: {y.shape}')
    
    return (X_h, X_d, X_w, y)
    
def generate_weather_data(file, hours, days, weeks, pred_window_size):
    
