
import math

def quantize_signal(signal_samples, bits=None, levels=None):
    if not isinstance(signal_samples, list) and not hasattr(signal_samples, 'dtype'):
        print("Error: signal_samples is not a list or numpy array.")
        return [], [], [], []
        
    if len(signal_samples) == 0:
        return [], [], [], []

    if bits is None and levels is None:
        raise ValueError("You must provide either 'bits' or 'levels'.")

    num_levels = 0
    num_bits = 0
    if bits is not None:
        num_bits = bits
        num_levels = 2**num_bits
    else:
        num_levels = levels
        if num_levels > 0:
            num_bits = int(math.ceil(math.log2(num_levels)))
        else:
            return [], [], [], []

    v_max = float(max(signal_samples))
    v_min = float(min(signal_samples))

    delta = (v_max - v_min) / num_levels if num_levels > 0 else 0
    
    if delta == 0:
        encoded_val = format(0, f'0{num_bits}b')
        num_samples = len(signal_samples)
        interval_indices = [0] * num_samples
        encoded_values = [encoded_val] * num_samples
        quantized_values = [float(s) for s in signal_samples]
        quantization_errors = [0.0] * num_samples
        return interval_indices, encoded_values, quantized_values, quantization_errors

    interval_indices = []
    encoded_values = []
    quantized_values = []
    quantization_errors = []

    for sample in signal_samples:
        j = math.floor((float(sample) - v_min) / delta)
        if j == num_levels:
            j = num_levels - 1
        v_q = v_min + (j + 0.5) * delta
        encoded = format(j, f'0{num_bits}b')
        error = v_q - float(sample)
        
        interval_indices.append(j)
        encoded_values.append(encoded)
        quantized_values.append(v_q)
        quantization_errors.append(error)

    return interval_indices, encoded_values, quantized_values, quantization_errors

def calculate_error_metrics(quantization_errors):
    if not quantization_errors:
        return 0.0, 0.0
    num_samples = len(quantization_errors)
    absolute_errors = [abs(e) for e in quantization_errors]
    mae = sum(absolute_errors) / num_samples
    squared_errors = [e**2 for e in quantization_errors]
    mse = sum(squared_errors) / num_samples
    return mae, mse