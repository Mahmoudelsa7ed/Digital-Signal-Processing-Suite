

import numpy as np
import os  
def read_signal(file_name):
    
    indices = []
    values = [] 
    folder_name = "TestCases"
    
    if os.path.exists(file_name):
        file_path = file_name
    else:
        file_path = os.path.join(folder_name, file_name)
        
    try:
        with open(file_path, 'r') as f:
            f.readline()  
            f.readline()  
            num_samples_line = f.readline().strip() 
            
            if not num_samples_line.isdigit():
                print(f"Error: Expected number of samples on line 3 in {file_name}")
                return np.array([]), np.array([])
                
            num_samples = int(num_samples_line)
            
            for _ in range(num_samples):
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        indices.append(int(parts[0]))
                        values.append(float(parts[1]))
                    
            if len(values) != num_samples:
                print(f"Warning: Mismatch between expected samples ({num_samples}) and read samples ({len(values)}) in {file_name}")

    except FileNotFoundError:
        print(f"Error: File not found! Looked at: {file_path}")
        print(f"Make sure the file is inside the '{folder_name}' folder.")
        return np.array([]), np.array([])
    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}")
        return np.array([]), np.array([])
            
    return np.array(indices), np.array(values)

def add_signals(signal1, signal2):
    indices1, values1 = signal1
    indices2, values2 = signal2
    all_indices = np.union1d(indices1, indices2)
    new_values1 = np.zeros_like(all_indices, dtype=float)
    new_values2 = np.zeros_like(all_indices, dtype=float)
    map_index1 = np.in1d(all_indices, indices1)
    new_values1[map_index1] = values1
    map_index2 = np.in1d(all_indices, indices2)
    new_values2[map_index2] = values2
    result_values = new_values1 + new_values2
    return all_indices, result_values

def subtract_signals(signal1, signal2):
    indices2, values2 = signal2
    neg_signal2 = (indices2, -1 * values2)
    return add_signals(signal1, neg_signal2)
    
def multiply_by_constant(signal, constant):
    indices, values = signal
    return indices, values * constant

def shift_signal(signal, k):
    indices, values = signal
    new_indices = indices + k
    return new_indices, values

def fold_signal(signal):
    indices, values = signal
    new_indices = -indices
    sorted_order = np.argsort(new_indices)
    return new_indices[sorted_order], values[sorted_order]

# task2
def generate_sinusoidal_signal(wave_type, amplitude, phase_shift, analog_freq, sampling_freq):
  
   
    num_samples = int(3 * (1 / analog_freq) * sampling_freq)
    n = np.arange(num_samples)
    if sampling_freq < 2 * analog_freq:
        
        return None, None


    phase_rad = np.deg2rad(phase_shift)
    
    digital_freq = 2 * np.pi * analog_freq / sampling_freq

    if wave_type == 'sine':
        values = amplitude * np.sin(digital_freq * n + phase_rad)
    elif wave_type == 'cosine':
        values = amplitude * np.cos(digital_freq * n + phase_rad)
    else:
        return None, None
        
    return n, values