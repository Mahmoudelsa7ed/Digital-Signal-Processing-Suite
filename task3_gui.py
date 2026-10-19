
import tkinter as tk
from tkinter import ttk, messagebox
from task3_quantization import quantize_signal, calculate_error_metrics

class QuantizationFrame(ttk.Labelframe):
    def __init__(self, parent_container, signals_dict):
        super().__init__(parent_container, text="Task 3: Quantization")
        self.loaded_signals = signals_dict 
        self._create_widgets()

    def _create_widgets(self):
        input_frame = ttk.Frame(self)
        input_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(input_frame, text="Select Signal:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.signal_quantize_combo = ttk.Combobox(input_frame, state="readonly")
        self.signal_quantize_combo.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        ttk.Label(input_frame, text="Number of Bits:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.bits_entry = ttk.Entry(input_frame, width=10)
        self.bits_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")

        ttk.Label(input_frame, text="Number of Levels:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.levels_entry = ttk.Entry(input_frame, width=10)
        self.levels_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")

        self.quantize_button = ttk.Button(input_frame, text="Run Quantization", command=self._perform_quantization)
        self.quantize_button.grid(row=3, column=0, columnspan=2, pady=10)

        output_frame = ttk.Frame(self)
        output_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        cols = ("Sample #", "Interval", "Encoded", "Quantized", "Error")
        self.quantize_table = ttk.Treeview(output_frame, columns=cols, show="headings")
        
        for col in cols:
            self.quantize_table.heading(col, text=col)
            
        self.quantize_table.column("Sample #", width=70, anchor="center")
        self.quantize_table.column("Interval", width=70, anchor="center")
        self.quantize_table.column("Encoded", width=80, anchor="center")
        self.quantize_table.column("Quantized", width=120, anchor="w") 
        self.quantize_table.column("Error", width=120, anchor="w") 
        
        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.quantize_table.yview)
        self.quantize_table.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.quantize_table.pack(side="left", fill="both", expand=True) 

        error_frame = ttk.Frame(self)
        error_frame.pack(fill="x", padx=5, pady=5)
        self.mae_label = ttk.Label(error_frame, text="Mean Absolute Error (MAE): -")
        self.mae_label.pack(side="left", padx=10)
        self.mse_label = ttk.Label(error_frame, text="Mean Square Error (MSE): -")
        self.mse_label.pack(side="left", padx=10)


    def _perform_quantization(self):
        
        def filter_digits(text_from_entry):
            return "".join(char for char in text_from_entry if char in "0123456789")

        signal_name = self.signal_quantize_combo.get()
        
        raw_bits = self.bits_entry.get()
        raw_levels = self.levels_entry.get()
        
        bits_str = filter_digits(raw_bits)
        levels_str = filter_digits(raw_levels)

        if not signal_name:
            messagebox.showerror("Error", "Please select a signal first.")
            return
            
        if bits_str and levels_str:
             messagebox.showerror("Error", "Please enter value in EITHER Bits OR Levels, not both.")
             return
             
        if not bits_str and not levels_str:
            messagebox.showerror("Error", "Please enter Number of Bits OR Number of Levels.")
            return

        try:
            signal_data = self.loaded_signals[signal_name] 
            
            samples = None
            if isinstance(signal_data, tuple):
                samples = list(signal_data[1])
            elif isinstance(signal_data, list):
                samples = signal_data
            elif hasattr(signal_data, 'dtype'):
                samples = list(signal_data[1])
            else:
                messagebox.showerror("Error", "Unknown signal data format in memory.")
                return

            if not samples:
                 messagebox.showerror("Error", "Signal contains no samples.")
                 return
                 
            quant_bits = None
            quant_levels = None
            
            if bits_str:
                quant_bits = int(bits_str)
            elif levels_str:
                quant_levels = int(levels_str)
            
        except (KeyError, ValueError, TypeError):
            messagebox.showerror("Error", "Invalid input. Bits/Levels must be numbers.")
            return
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

        indices, encoded, quantized, errors = quantize_signal(
            signal_samples=samples, bits=quant_bits, levels=quant_levels
        )
        mae, mse = calculate_error_metrics(errors)
        
        self.quantize_table.delete(*self.quantize_table.get_children())
        for i in range(len(samples)):
            row_data = (i, indices[i], encoded[i], f"{quantized[i]:.4f}", f"{errors[i]:.4f}")
            self.quantize_table.insert("", "end", values=row_data)
            
        self.mae_label.config(text=f"Mean Absolute Error (MAE): {mae:.6f}")
        self.mse_label.config(text=f"Mean Square Error (MSE): {mse:.6f}")
        
    def update_signal_list(self):
        self.signal_quantize_combo['values'] = list(self.loaded_signals.keys())
        if list(self.loaded_signals.keys()):
            self.signal_quantize_combo.set(list(self.loaded_signals.keys())[-1])