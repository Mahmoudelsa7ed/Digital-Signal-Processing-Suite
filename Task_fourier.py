import numpy as np
import cmath

def discrete_fourier_transform(signal_values, fs, is_inverse=False):
    
    N = len(signal_values)
    result = []
    
    
    exponent_sign = 1 if is_inverse else -1
    
    
    for k in range(N):
        sum_val = 0
        for n in range(N):
            sample = signal_values[n]
            
            #  (2 * pi * k * n / N)
            angle = 2 * np.pi * k * n / N
            
            #  e^(sign * j * angle)
            harmonic = np.exp(1j * exponent_sign * angle)
            
            sum_val += sample * harmonic
            
        if is_inverse:
            sum_val /= N
            
        result.append(sum_val)
        
    result_complex = np.array(result)

    #  (Post-Processing)
    if is_inverse:
        indices = np.arange(N)
        reconstructed_values = np.real(result_complex)
        return indices, reconstructed_values
    
    else:
        # f = k * Fs / N
        frequencies = np.array([k * fs / N for k in range(N)])
        
       
        amplitudes = np.abs(result_complex)
        
        
        phases = np.angle(result_complex)
        
        
        return frequencies, amplitudes, phases, result_complex