import os
import numpy as np

def read_raw_signal(file_path):
 
    samples = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                val = line.strip()
                if val: 
                    samples.append(float(val))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    values = np.array(samples)
    indices = np.arange(len(values))
    return indices, values
def fast_correlation(values1, values2):
    n1 = len(values1)
    n2 = len(values2)
    N = n1 + n2 - 1 # Linear Convolution
    
    # Zero-Padding
    fft_v1 = np.fft.fft(values1, N)
    fft_v2 = np.fft.fft(values2, N)
    
    #  Conjugate * FFT
    corr_freq = np.conj(fft_v1) * fft_v2
    
    corr_values = np.real(np.fft.ifft(corr_freq))
    
    # Normalization
    energy1 = np.sum(values1 ** 2)
    energy2 = np.sum(values2 ** 2)
    norm_factor = np.sqrt(energy1 * energy2)
    
    if norm_factor == 0: norm_factor = 1
    
    normalized_corr = corr_values / norm_factor
    
    return normalized_corr

#  (Classify Logic)
def classify_signal(test_file_path, class1_folder, class2_folder):
    print(f"\n--- Classifying: {test_file_path} ---")
    
    _, test_samples = read_raw_signal(test_file_path)
    if len(test_samples) == 0:
        print("Error: Test signal is empty.")
        return

    def get_avg_max_corr(folder_path, test_sig):
        max_corrs = []
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        if not files:
            print(f"Warning: Folder {folder_path} is empty!")
            return 0
            
        print(f"Comparing with {len(files)} files in {folder_path}...")
        
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            _, train_samples = read_raw_signal(filepath)
            
            corr_result = fast_correlation(test_sig, train_samples)
            
            #  (Max Correlation)
            max_val = np.max(np.abs(corr_result))
            max_corrs.append(max_val)
            
        return np.mean(max_corrs)

    #  Class 1 (Down)
    avg_class1 = get_avg_max_corr(class1_folder, test_samples)
    
    #  Class 2 (Up)
    avg_class2 = get_avg_max_corr(class2_folder, test_samples)
    
    print(f"Avg Correlation with Class 1 (Down): {avg_class1:.5f}")
    print(f"Avg Correlation with Class 2 (Up)  : {avg_class2:.5f}")
    
    if avg_class1 > avg_class2:
        print(">>> Result: Class 1 (Down)")
        return "Class 1"
    else:
        print(">>> Result: Class 2 (Up)")
        return "Class 2"

if __name__ == "__main__":
    path_class1 = "Class1"  
    path_class2 = "Class2" 
    
    test_files = ["Test1.txt", "Test2.txt"]
    
    if os.path.exists(path_class1) and os.path.exists(path_class2):
        for t_file in test_files:
            if os.path.exists(t_file):
                classify_signal(t_file, path_class1, path_class2)
            else:
                print(f"File {t_file} not found.")
    else:
        print("Please create folders 'Class1' and 'Class2' and put the text files inside them.")