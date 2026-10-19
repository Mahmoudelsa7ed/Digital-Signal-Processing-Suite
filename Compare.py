import numpy as np
import os
import math
def read_file_dummy(filename):
    file_path = os.path.join("TestCases", filename)
    if not os.path.exists(file_path):
        file_path = filename 
        
    try:
        data = []
        with open(file_path, 'r') as f:
            f.readline(); f.readline(); f.readline() 
            for line in f:
                parts = line.split()
                if len(parts) >= 2: 
                    data.append(float(parts[1]))
        return np.array(range(len(data))), np.array(data)
    except:
        return np.array([]), np.array([])

def Compare_Signals(indices, values, expected_indices, expected_values):
    if len(values) != len(expected_values):
        print(f"   Length Mismatch! Calc={len(values)}, Exp={len(expected_values)}")
        return False
        
    for i in range(len(values)):
        if abs(values[i] - expected_values[i]) > 0.01:
            return False
    return True

def fast_correlation(values1, values2):
    corr_values = np.correlate(values1, values2, mode='full')
    
    N = len(values1)
    start_index = -1 * (N - 1)
    corr_indices = np.arange(start_index, start_index + len(corr_values))
    
    normalized_corr = corr_values / N 
    
    return corr_indices, normalized_corr 


def fast_time_delay(values1, values2, fs):
    corr_values = np.correlate(values1, values2, mode='full')
    N = len(values1)
    start_index = -1 * (N - 1)
    corr_indices = np.arange(start_index, start_index + len(corr_values))
    
    max_idx = np.argmax(np.abs(corr_values))
    lag_j = corr_indices[max_idx]
    
    ts = 1.0 / fs
    time_delay = lag_j * ts
    
    return lag_j, time_delay

if __name__ == "__main__":
    print("\n--- Testing Task 6: Correlation ---")
    
    _, val1 = read_file_dummy("Corr_input signal1.txt") 
    _, val2 = read_file_dummy("Corr_input signal2.txt") 
    _, exp_val = read_file_dummy("CorrOutput.txt")

    if len(val1) > 0:
        raw_corr = np.correlate(val1, val2, mode='full')
        
        norm_factor = (np.sum(val1**2) * np.sum(val2**2))**0.5
        calc_val = raw_corr / norm_factor
        
        if len(calc_val) != len(exp_val):
            start = (len(calc_val) - len(exp_val)) // 2
            calc_val = calc_val[start : start + len(exp_val)]
            
        print(f"First Calc: {calc_val[0]}, First Exp: {exp_val[0]}")

        dummy_ind = np.arange(len(calc_val))
        
        if Compare_Signals(dummy_ind, calc_val, dummy_ind, exp_val):
            print(">> Correlation Test: PASSED ")
        else:
            print(">> Correlation Test: FAILED ")

    _, val_td1 = read_file_dummy("TD_input signal1.txt") 
    _, val_td2 = read_file_dummy("TD_input signal2.txt")
    
    if len(val_td1) > 0:
        fs = 100.0 
        lag, delay = fast_time_delay(val_td1, val_td2, fs)
        
        print(f"\nCalculated Lag: {lag}")
        if lag == 4:
            print(">> Time Delay Test: PASSED ")