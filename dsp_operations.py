
import numpy as np
import Compare as comp
import Task_fourier as tf
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
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline() 
            
            try:
                num_samples = int(line3.strip())
            except ValueError:
                num_samples = 0 

            for line in f:
                clean_line = line.replace(',', ' ').replace('f', '').strip()
                
                if not clean_line: continue 
                
                parts = clean_line.split()
                
                if len(parts) >= 2:
                    indices.append(int(float(parts[0]))) 
                    values.append(float(parts[1]))
                    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return np.array([]), np.array([])
    except Exception as e:
        print(f"An error occurred while reading {file_name}: {e}")
        return np.array([]), np.array([])
            
    return np.array(indices), np.array(values)
# --- دوال TASK 5  ---

def moving_average(indices, values, M):
    
    if M <= 0:
        print("Error: Window size must be greater than 0")
        return indices, values 
    if M > len(values):
        print("Warning: Window size is larger than signal length")
        return np.array([]), np.array([]) 
    window = np.ones(M) / M
    
    y_values = np.convolve(values, window, mode='valid')
    
    y_indices = indices[:len(y_values)]
    
    return y_indices, y_values

def first_derivative(indices, values):
    """
    (First Derivative).
    Y(n) = x(n) - x(n-1)
    """
    y_values = np.diff(values)
    
    y_indices = indices[:-1]
    
    return y_indices, y_values

def second_derivative(indices, values):
    """
   (Second Derivative).
    Y(n) = x(n+1) - 2x(n) + x(n-1)
    """
    window = np.array([1, -2, 1])
    
    
    y_values = np.convolve(values, window, mode='valid')
    
    y_indices = indices[:len(y_values)] # (أو indices[:-2])
    
    return y_indices, y_values

def convolution(indices1, values1, indices2, values2):
    y_values = np.convolve(values1, values2, mode='full')
    
    start_index = 0
    if len(indices1) > 0 and len(indices2) > 0:
        start_index = indices1[0] + indices2[0]
    
    num_samples_out = len(y_values)
    y_indices = np.arange(start_index, start_index + num_samples_out)
    
    return y_indices, y_values

def CompareSignals(your_indices, your_values, expected_filename, tolerance=0.01):
    
    print(f"--- Comparing with {expected_filename} ---")
    
    
    expected_indices, expected_values = read_signal(expected_filename)
    
    if expected_values.size == 0:
        print(f"ERROR: Could not read expected file: {expected_filename}")
        print("Test FAILED ")
        return

    if len(your_values) != len(expected_values):
        print(f"FAILED: Mismatch in number of samples.")
        print(f"  Expected {len(expected_values)} samples, but you produced {len(your_values)} samples.")
        print("Test FAILED ")
        return

    if not np.array_equal(your_indices, expected_indices):
        print(f"FAILED: Mismatch in indices.")
        print(f"  Expected indices start: {expected_indices[0] if len(expected_indices) > 0 else 'N/A'}")
        print(f"  Your indices start: {your_indices[0] if len(your_indices) > 0 else 'N/A'}")
        print(f"  Expected: {expected_indices}")
        print(f"  Yours:    {your_indices}")
        print("Test FAILED ")
        return

    if not np.allclose(your_values, expected_values, atol=tolerance):
        print(f"FAILED: Mismatch in values (beyond tolerance={tolerance}).")
        for i in range(len(your_values)):
            if not np.isclose(your_values[i], expected_values[i], atol=tolerance):
                print(f"  at index {your_indices[i]}: Expected={expected_values[i]}, Yours={your_values[i]}")
                break
        print("Test FAILED ")
        return

    print(f"Test PASSED ")
if __name__ == "__main__":
    
    print("--- Running DSP Operations Tests ---")
    
    M_test1 = 3
    indices_ma, values_ma = read_signal("MovingAvg_input.txt")
    if values_ma.size > 0:
        out_indices_ma1, out_values_ma1 = moving_average(indices_ma, values_ma, M_test1)
        print("\n--- Moving Average (M=3) ---")
        print("Output Values:", np.round(out_values_ma1, 3))
        CompareSignals(out_indices_ma1, out_values_ma1, "MovingAvg_out1.txt")

    M_test2 = 5
    if values_ma.size > 0:
        out_indices_ma2, out_values_ma2 = moving_average(indices_ma, values_ma, M_test2)
        print("\n--- Moving Average (M=5) ---")
        print("Output Values:", np.round(out_values_ma2, 3))
        CompareSignals(out_indices_ma2, out_values_ma2, "MovingAvg_out2.txt")

    indices_d, values_d = read_signal("Derivative_input.txt")
    if values_d.size > 0:
        out_indices_d1, out_values_d1 = first_derivative(indices_d, values_d)
        print("\n--- First Derivative ---")
        print("Output Values:", out_values_d1)
        CompareSignals(out_indices_d1, out_values_d1, "1st_derivative_out.txt")

    if values_d.size > 0:
        out_indices_d2, out_values_d2 = second_derivative(indices_d, values_d)
        print("\n--- Second Derivative ---")
        print("Output Values:", out_values_d2)
        CompareSignals(out_indices_d2, out_values_d2, "2nd_derivative_out.txt")

    indices_c1, values_c1 = read_signal("Signal 1.txt")
    indices_c2, values_c2 = read_signal("Signal 2.txt")
    
    if values_c1.size > 0 and values_c2.size > 0:
        out_indices_conv, out_values_conv = convolution(indices_c1, values_c1, indices_c2, values_c2)
        print("\n--- Convolution ---")
        print("Output Indices:", out_indices_conv)
        print("Output Values:", out_values_conv)
        CompareSignals(out_indices_conv, out_values_conv, "Conv_output.txt")
 
if __name__ == "__main__":
    
    print("\n" + "="*40)
   
    print("="*40)

    input_filename = "input_Signal_DFT.txt"
    indices, values = read_signal(input_filename)
    
    if len(values) == 0:
        print(f" Error: Empty input signal from {input_filename}")
    else:
        print(f"file: {input_filename}")
        print(f"First 5 Input Values read: {values[:5]}") 
        fs = 1 
        freqs, calculated_amps, calculated_phases, _ = tf.discrete_fourier_transform(values, fs, is_inverse=False)
        
        expected_amps = []
        expected_phases = []
        
        output_filename = "Output_Signal_DFT_A,Phase.txt"
        file_path = os.path.join("TestCases", output_filename)
        if not os.path.exists(file_path): file_path = output_filename

        try:
            with open(file_path, 'r') as f:
                f.readline(); f.readline(); f.readline()
                
                for line in f:
                    clean_line = line.replace(',', ' ').replace('f', '').strip()
                    parts = clean_line.split()
                    
                    if len(parts) >= 3: 
                        expected_amps.append(float(parts[1]))   
                        expected_phases.append(float(parts[2])) 
                        
                    elif len(parts) == 2:
                        expected_amps.append(float(parts[0]))
                        expected_phases.append(float(parts[1]))

            print("\n[Comparison Results]")
            
            # Amplitude
            print(f"First Expected Amp: {expected_amps[0] if expected_amps else 'None'}")
            print(f"First Calc Amp:     {calculated_amps[0]}")
            
            if comp.SignalCompareAmplitude(calculated_amps, expected_amps):
                print(">> Amplitude Test: PASSED ")
            else:
                print(">> Amplitude Test: FAILED ")

            print(f"First Expected Phase: {expected_phases[0] if expected_phases else 'None'}")
            print(f"First Calc Phase:     {calculated_phases[0]}")

            if len(expected_phases) > 0:
                if comp.SignalComparePhaseShift(calculated_phases, expected_phases):
                    print(">> Phase Test: PASSED ")
                else:
                    print(">> Phase Test: FAILED ")
            else:
                print(">> Phase Test: Skipped (No data read from output file)")

        except FileNotFoundError:
            print(f"Error: Could not find '{output_filename}'")
        except ValueError as e:
            print(f"Error parsing output file: {e}")