import numpy as np
import math
import os
import matplotlib.pyplot as plt

# 1. Helper Functions (Reading & Comparing)
def read_specs(file_path):
    specs = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=')
                    specs[k.strip()] = v.strip()
    except: return {}
    return specs

def read_signal_file(file_path):
    vals = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    try: vals.append(float(parts[1]))
                    except: continue
    except: return []
    return np.array(vals)

def get_expected_info(file_path):
    indices = []
    values = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    indices.append(int(parts[0]))
                    values.append(float(parts[1]))
    except: return [], []
    return indices, values

def Compare_Signals(file_name, Your_indices, Your_samples):      
    exp_indices, exp_samples = get_expected_info(file_name)
    if len(exp_samples) == 0:
        print(f"   Warning: Expected file missing/empty.")
        return

    if len(exp_samples) != len(Your_samples):
        print(f"   Test failed: Length Mismatch. Exp:{len(exp_samples)}, Got:{len(Your_samples)}")
        return

    for i in range(len(Your_samples)):
        if abs(Your_samples[i] - exp_samples[i]) > 0.01:
            curr_idx = Your_indices[i] if len(Your_indices) > i else i
            print(f"  Test failed at index {curr_idx}: Exp={exp_samples[i]}, Got={Your_samples[i]}") 
            return
            
    print("   Test Case Passed Successfully")

# =============================================================================
# 2. FIR Design Logic
# =============================================================================
def get_window_type_and_N(stop_band_attenuation, transition_band, fs):
    delta_f = transition_band / fs 
    window_type = ""
    N = 0
    
    if stop_band_attenuation <= 21:
        window_type = "rectangular"; N = 0.9 / delta_f
    elif stop_band_attenuation <= 44:
        window_type = "hanning"; N = 3.1 / delta_f
    elif stop_band_attenuation <= 53:
        window_type = "hamming"; N = 3.3 / delta_f
    else:
        window_type = "blackman"; N = 5.5 / delta_f
        
    N = int(math.ceil(N))
    if N % 2 == 0: N += 1
    return N, window_type

def design_fir_coefficients(filter_type, fs, stop_attenuation, trans_band, f1, f2=None):
    N, window_name = get_window_type_and_N(stop_attenuation, trans_band, fs)
    fc1_new, fc2_new = 0, 0
    half_trans = trans_band / 2
    ftype = filter_type.lower().replace("-", " ").strip()

    if "low" in ftype: fc1_new = (f1 + half_trans) / fs
    elif "high" in ftype: fc1_new = (f1 - half_trans) / fs
    elif "band pass" in ftype:
        fc1_new = (f1 - half_trans) / fs; fc2_new = (f2 + half_trans) / fs
    elif "band stop" in ftype:
        fc1_new = (f1 + half_trans) / fs; fc2_new = (f2 - half_trans) / fs

    coefficients = []
    alpha = (N - 1) / 2
    
    for i in range(N):
        n = i - alpha
        hd = 0
        w1 = 2 * math.pi * fc1_new
        w2 = 2 * math.pi * fc2_new if f2 else 0
        
        if n == 0:
            if "low" in ftype: hd = 2 * fc1_new
            elif "high" in ftype: hd = 1 - (2 * fc1_new)
            elif "band pass" in ftype: hd = 2 * (fc2_new - fc1_new)
            elif "band stop" in ftype: hd = 1 - (2 * (fc2_new - fc1_new))
        else:
            if "low" in ftype: hd = 2 * fc1_new * (math.sin(n * w1) / (n * w1))
            elif "high" in ftype: hd = -2 * fc1_new * (math.sin(n * w1) / (n * w1))
            elif "band pass" in ftype:
                hd = (2*fc2_new*(math.sin(n*w2)/(n*w2))) - (2*fc1_new*(math.sin(n*w1)/(n*w1)))
            elif "band stop" in ftype:
                hd = (2*fc1_new*(math.sin(n*w1)/(n*w1))) - (2*fc2_new*(math.sin(n*w2)/(n*w2)))

        wn = 1.0
        if window_name == "hanning": wn = 0.5 + 0.5 * math.cos((2*math.pi*n)/N)
        elif window_name == "hamming": wn = 0.54 + 0.46 * math.cos((2*math.pi*n)/N)
        elif window_name == "blackman":
            wn = 0.42 + 0.5*math.cos((2*math.pi*n)/(N-1)) + 0.08*math.cos((4*math.pi*n)/(N-1))
        
        coefficients.append(hd * wn)
    return np.array(coefficients)

def apply_filter_direct(input_signal, coefficients):
    N_sig = len(input_signal)
    M_coeff = len(coefficients)
    Total_Len = N_sig + M_coeff - 1
    result = np.zeros(Total_Len)
    for i in range(Total_Len):
        for j in range(M_coeff):
            if (i - j) >= 0 and (i - j) < N_sig:
                result[i] += input_signal[i - j] * coefficients[j]
    return result

# =============================================================================
# 4. GUI Support Functions 
# =============================================================================

def save_coefficients_to_file(file_name, coeffs):
    try:
        N = len(coeffs)
        start_index = -int((N-1)/2)
        
        with open(file_name, 'w') as f:
            f.write("0\n") 
            f.write("0\n") 
            f.write(f"{N}\n")
            
            
            for i, val in enumerate(coeffs):
                current_idx = start_index + i
                formatted_val = str(val).upper()
                f.write(f"{current_idx} {formatted_val}\n") 
                
    except Exception as e:
        print(f"Error saving coefficients: {e}")
        
def apply_filter_fast(input_signal, coefficients):
    """Convolution using FFT (Fast Convolution)."""
    # N = L + M - 1
    N_fft = len(input_signal) + len(coefficients) - 1
    
    # Compute FFT
    X = np.fft.fft(input_signal, N_fft)
    H = np.fft.fft(coefficients, N_fft)
    
    # Multiply in Frequency Domain
    Y = X * H
    
    # Inverse FFT
    y = np.fft.ifft(Y)
    
    # Return Real part
    return np.real(y)

def save_signal_to_file(file_name, indices, samples):
    try:
        with open(file_name, 'w') as f:
            f.write("0\n")  
            f.write("0\n")  
            f.write(f"{len(samples)}\n") 
            
            for i in range(len(samples)):
                val = samples[i]
                formatted_val = str(val).upper()
                
                f.write(f"{int(indices[i])} {formatted_val}\n")
                
        print(f"Saved signal to {file_name}")
    except Exception as e:
        print(f"Error saving signal: {e}")
def compare_with_file(file_path, your_samples):
    exp_indices, exp_samples = get_expected_info(file_path)
    
    if len(exp_samples) == 0:
        return "Error: Expected file not found or empty."
        
    min_len = min(len(exp_samples), len(your_samples))
    
    mismatches = []
    for i in range(min_len):
        if abs(your_samples[i] - exp_samples[i]) > 0.01:
            mismatches.append(f"Index {i}: Exp={exp_samples[i]}, Got={your_samples[i]}")
            if len(mismatches) > 3: break # Stop after few errors
            
    if not mismatches:
        return "Test Passed Successfully!"
    else:
        return "Test Failed.\nSamples:\n" + "\n".join(mismatches)

# =============================================================================
# 3. MAIN RUNNER (With Info Printing & Plots)
# =============================================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running FIR tests from: {current_dir}\n")
    
    SHOW_PLOTS = True 
    
    for i in range(1, 9):
        TEST_FOLDER = f"Testcase {i}"
        base_path = os.path.join(current_dir, TEST_FOLDER)
        
        spec_file = os.path.join(base_path, "Filter Specifications.txt")
        if not os.path.exists(spec_file):
            continue
            
        print(f"\n{'='*60}")
        print(f" >>> {TEST_FOLDER}")
        print(f"{'='*60}")

        specs = read_specs(spec_file)
        ftype = specs.get("FilterType")
        fs = float(specs.get("FS"))
        stop_att = float(specs.get("StopBandAttenuation"))
        trans = float(specs.get("TransitionBand"))
        f1 = float(specs.get("F1") if specs.get("F1") else specs.get("FC"))
        f2 = float(specs.get("F2") if specs.get("F2") else 0)
        
        N_calc, win_name = get_window_type_and_N(stop_att, trans, fs)
        
        print(f"    [Filter Info]")
        print(f"      • Type        : {ftype}")
        print(f"      • Sampling Fs : {fs} Hz")
        print(f"      • Window Used : {win_name.upper()}")
        print(f"      • Order (N)   : {N_calc}")
        if f2 != 0:
            print(f"      • Cutoffs     : F1={f1} Hz, F2={f2} Hz")
        else:
            print(f"      • Cutoff      : Fc={f1} Hz")
        print(f"      • Stop Atten. : {stop_att} dB")
        print("-" * 40)

        coeffs = design_fir_coefficients(ftype, fs, stop_att, trans, f1, f2)
        
        # ---------------------------------------------------------
        # A. اختبار المعاملات (Test 1,3,5,7)
        # ---------------------------------------------------------
        coeff_file = ""
        if "low" in ftype.lower(): coeff_file = "LPFCoefficients.txt"
        elif "high" in ftype.lower(): coeff_file = "HPFCoefficients.txt"
        elif "band pass" in ftype.lower(): coeff_file = "BPFCoefficients.txt"
        elif "band stop" in ftype.lower(): coeff_file = "BSFCoefficients.txt"
        
        c_path = os.path.join(base_path, coeff_file)
        if os.path.exists(c_path):
            print(f"   🔍 [Checking Coefficients]")
            N = len(coeffs)
            start_index = -int((N-1)/2)
            my_indices = np.arange(start_index, start_index + N)
            Compare_Signals(c_path, my_indices, coeffs)
            
            # رسم الـ Impulse Response
            if SHOW_PLOTS:
                plt.figure(figsize=(8, 4))
                plt.stem(my_indices, coeffs)
                plt.title(f"{TEST_FOLDER}: {ftype} - Impulse Response h(n)\nN={N_calc}, Window={win_name}")
                plt.xlabel("n")
                plt.ylabel("Amplitude")
                plt.grid(True)
                plt.show()

        # ---------------------------------------------------------
        # B. اختبار الإشارة (Test 2,4,6,8)
        # ---------------------------------------------------------
        sig_path = os.path.join(base_path, "ecg400.txt")
        if os.path.exists(sig_path):
            print(f"   🔍 [Checking Filtering Output]")
            sig = read_signal_file(sig_path)
            if len(sig) == 0: continue

            filtered = apply_filter_direct(sig, coeffs)
            
            out_file = ""
            if "low" in ftype.lower(): out_file = "ecg_low_pass_filtered.txt"
            elif "high" in ftype.lower(): out_file = "ecg_high_pass_filtered.txt"
            elif "band pass" in ftype.lower(): out_file = "ecg_band_pass_filtered.txt"
            elif "band stop" in ftype.lower(): out_file = "ecg_band_stop_filtered.txt"
            
            o_path = os.path.join(base_path, out_file)
            if os.path.exists(o_path):
                N = len(coeffs)
                filter_start_index = -int((N-1)/2)
                my_indices_sig = np.arange(filter_start_index, filter_start_index + len(filtered))
                
                Compare_Signals(o_path, my_indices_sig, filtered)

                if SHOW_PLOTS:
                    plt.figure(figsize=(10, 6))
                    
                    plt.subplot(2, 1, 1)
                    plt.plot(sig, 'b')
                    plt.title(f"{TEST_FOLDER}: Original ECG Signal")
                    plt.grid(True)
                    
                    plt.subplot(2, 1, 2)
                    plt.plot(my_indices_sig, filtered, 'r')
                    plt.title(f"Filtered Signal ({ftype})")
                    plt.xlabel("n (Time Index)")
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.show()

    print("\n All Tests Completed.")
    input("Press Enter to exit...")