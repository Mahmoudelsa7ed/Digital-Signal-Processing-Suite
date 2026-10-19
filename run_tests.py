

import numpy as np

try:
    from QuanTest1 import QuantizationTest1
    from QuanTest2 import QuantizationTest2
except ImportError:
    print("wrong")
    exit()

try:
    from task3_quantization import quantize_signal, calculate_error_metrics
except ImportError:
    print("wrong")
    exit()

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

print("=========Test Quantization =========\n")

print("---  (3 bits) ---")

samples1 = read_signal('Quan1_input.txt')

if samples1:
    try:
        indices_1, encoded_1, quantized_1, errors_1 = quantize_signal(samples1, bits=3)

        quantized_1_rounded = [round(q, 5) for q in quantized_1]

        QuantizationTest1("Quan1_Out.txt", encoded_1, quantized_1_rounded)

        mae_1, mse_1 = calculate_error_metrics(errors_1)
        print(f"Test Case 1 - MAE: {mae_1:.6f}, MSE: {mse_1:.6f}")

    except Exception as e:
        print(f":wrong{e}")


print("\n--- (4 levels) ---")

samples2 = read_signal('Quan2_input.txt')

if samples2:
    try:
        indices_2, encoded_2, quantized_2, errors_2 = quantize_signal(samples2, levels=4)

        quantized_2_rounded = [round(q, 5) for q in quantized_2]
        errors_2_rounded = [round(e, 5) for e in errors_2]

        QuantizationTest2("Quan2_Out.txt", 
                          indices_2, 
                          encoded_2, 
                          quantized_2_rounded, 
                          errors_2_rounded)

        mae_2, mse_2 = calculate_error_metrics(errors_2)
        print(f"Test Case 2 - MAE: {mae_2:.6f}, MSE: {mse_2:.6f}")

    except Exception as e:
        print(f"wrong{e}")

print("\n========= The End ==========")