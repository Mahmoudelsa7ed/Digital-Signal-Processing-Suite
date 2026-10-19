import numpy as np
import math
import os
import matplotlib.pyplot as plt

def read_file_values(file_path):
    indices = []
    vals = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        idx = int(parts[0])
                        val = float(parts[1])
                        indices.append(idx)
                        vals.append(val)
                    except ValueError: continue
                elif len(parts) == 1:
                    try:
                        val = float(parts[0])
                        vals.append(val)
                    except ValueError: continue
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    if not vals: return np.array([])
    if len(indices) > 0:
        start_index = indices[0]
        if start_index > 0:
            padding = [0.0] * start_index
            vals = padding + vals
            
    return np.array(vals)

# ====================================================================
# 1. Direct Correlation (Task 1)
# ====================================================================
def direct_correlation(indices1, values1, indices2, values2):
    values1 = np.array(values1)
    values2 = np.array(values2)
    N = len(values1)
    
    if len(values2) != N:
        min_len = min(len(values1), len(values2))
        values1 = values1[:min_len]
        values2 = values2[:min_len]
        N = min_len

    r12 = [] 
    for j in range(N):
        current_sum = 0
        for n in range(N):
            shift_idx = (n + j) % N 
            current_sum += values1[n] * values2[shift_idx]
        r12.append(current_sum / N)
        
    r12 = np.array(r12)

    # Normalization
    energy1 = np.sum(values1 ** 2) / N
    energy2 = np.sum(values2 ** 2) / N
    norm_factor = math.sqrt(energy1 * energy2)
    if norm_factor == 0: norm_factor = 1
    
    normalized_r12 = r12 / norm_factor
    positive_corr_indices = np.arange(0, N)
    return positive_corr_indices, normalized_r12

def fast_correlation(indices1, values1, indices2, values2):
    return direct_correlation(indices1, values1, indices2, values2)

# ====================================================================
# 2. Time Delay (Task 2)
# ====================================================================
def fast_time_delay(indices1, values1, indices2, values2, fs):
    _, corr_values = direct_correlation(indices1, values1, indices2, values2)
    
    max_idx = np.argmax(np.abs(corr_values))
    lag_j = max_idx
    
    if fs == 0: fs = 1
    Ts = 1.0 / fs
    time_delay = lag_j * Ts
    
    return lag_j, time_delay

# ====================================================================
# 3. Classification (Task 3)
# ====================================================================
def classify_signal(test_file, class1_folder, class2_folder):
    print(f"\n--- Classifying Signal: {os.path.basename(test_file)} ---")
    test_sig = read_file_values(test_file)
    
    if len(test_sig) == 0: 
        print("Error: Test signal empty or not found.")
        return

    def get_avg_max_corr(folder_path, input_sig):
        max_correlations = []
        if not os.path.exists(folder_path): return 0
        
        files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        if not files: return 0
        
        for fname in files:
            path = os.path.join(folder_path, fname)
            class_sig = read_file_values(path)
            
            # توحيد الطول
            L = min(len(input_sig), len(class_sig))
            if L == 0: continue
            
            sig1 = input_sig[:L]
            sig2 = class_sig[:L]
            ind = np.arange(L)
            
            # حساب Direct Correlation
            _, corr_out = direct_correlation(ind, sig1, ind, sig2)
            max_corrs = np.max(np.abs(corr_out))
            max_correlations.append(max_corrs)
            
        return np.mean(max_correlations) if max_correlations else 0

    avg_A = get_avg_max_corr(class1_folder, test_sig)
    avg_B = get_avg_max_corr(class2_folder, test_sig)
    
    print(f"Avg Max Correlation with Class 1 (Down): {avg_A}")
    print(f"Avg Max Correlation with Class 2 (Up)  : {avg_B}")
    
    if avg_A > avg_B:
        print(">>> Result: Signal belongs to Class 1 (Down)")
    else:
        print(">>> Result: Signal belongs to Class 2 (Up)")

# ====================================================================
# Compare Function
# ====================================================================
def Compare_Signals(file_name, Your_indices, Your_samples):      
    expected_indices=[]
    expected_samples=[]
    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            start_idx = 0
            for i, line in enumerate(lines):
                if len(line.split()) == 2:
                    start_idx = i
                    break
            for i in range(start_idx, len(lines)):
                L = lines[i].strip().split()
                if len(L)==2:
                    expected_indices.append(int(L[0]))
                    expected_samples.append(float(L[1]))
    except: return

    print(f"Verifying with {os.path.basename(file_name)} ...")
    if (len(expected_samples)!=len(Your_samples)):
        print(f"Test failed: Length Mismatch. Exp:{len(expected_samples)}, Got:{len(Your_samples)}")
        return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print(f"Test failed at {i}: Exp={expected_samples[i]}, Got={Your_samples[i]}") 
            return
    print("Correlation Test case passed successfully")

# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    
    # 1. Direct Correlation
    print("--- 1. Testing Direct Correlation ---")
    x1 = [2, 1, 0, 0, 3]
    x2 = [3, 2, 1, 1, 5] 
    idx = [0, 1, 2, 3, 4]
    ind, val = direct_correlation(idx, x1, idx, x2)
    print("Direct Output:", val)
    if os.path.exists("TestCases/CorrOutput.txt"):
        Compare_Signals("TestCases/CorrOutput.txt", ind, val)

    # 2. Time Delay
    print("\n--- 2. Testing Time Delay ---")
    if os.path.exists("TestCases/TD_input signal1.txt"):
        sig1 = read_file_values("TestCases/TD_input signal1.txt")
        sig2 = read_file_values("TestCases/TD_input signal2.txt")
        Fs = 100
        fake_ind = np.arange(len(sig1))
        
        lag, delay = fast_time_delay(fake_ind, sig1, fake_ind, sig2, Fs)
        print(f"Lag: {lag} samples")
        print(f"Delay: {delay} seconds") 

    # 3. Classification
    print("\n--- 3. Testing Classification ---")
    if os.path.exists("Class1") and os.path.exists("Class2"):
        classify_signal("Test1.txt", "Class1", "Class2")
        classify_signal("Test2.txt", "Class1", "Class2")

    # 4. Plot
    print("\nDisplaying Plot... (Close window to finish)")
    plt.stem(ind, val)
    plt.title("Direct Correlation Output")
    plt.grid(True)
    plt.show()