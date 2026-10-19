
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import dsp_functions as dsp 
from task3_gui import QuantizationFrame
import os
import numpy as np
import Task_fourier as tf
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import FIR_Filter as fir  
import matplotlib.pyplot as plt 

from dsp_operations import (
    read_signal,  
    moving_average, 
    first_derivative, 
    second_derivative, 
    convolution
)
import correlation as corr

class SignalProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSP Project (Tasks 1, 2, 3 & 5)")
        self.geometry("1300x800")

        self.loaded_signals = {}
        self.generated_signal_count = 0
        
        self.last_dft_complex_result = None 
        
        self._create_widgets()

    def _create_widgets(self):
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # --- 1. الجزء الأيمن (الرسم البياني) ---
        right_pane = ttk.Frame(main_pane)
        main_pane.add(right_pane)

        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_pane)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right_pane)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        # --- 2. الجزء الأيسر (لوحة التحكم القابلة للتمرير) ---
        left_container = ttk.Frame(main_pane, width=500)
        main_pane.add(left_container, minsize=480)
        left_container.pack_propagate(False)

        canvas = tk.Canvas(left_container)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.scrollable_frame.bind("<Configure>", on_frame_configure)
        
        def on_mouse_wheel(event):
            delta = -1 * (event.delta // 120) if os.name == 'nt' else event.delta
            canvas.yview_scroll(delta, "units")
        self.bind_all("<MouseWheel>", on_mouse_wheel)

        # --- محتويات الجزء الأيسر (داخل self.scrollable_frame) ---
        
        # قسم 1: تحميل الإشارات
        load_frame = ttk.Labelframe(self.scrollable_frame, text="1. Load / Generate Signals")
        load_frame.pack(fill="x", padx=10, pady=5)
        # (باقي المحتويات)
        load_button = ttk.Button(load_frame, text="Load Signal from .txt File", command=self._load_signal)
        load_button.pack(pady=5, fill="x", padx=5)
        gen_frame = ttk.Frame(load_frame)
        gen_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(gen_frame, text="Amplitude:").grid(row=0, column=0, sticky="w")
        self.amp_entry = ttk.Entry(gen_frame, width=10)
        self.amp_entry.grid(row=0, column=1, padx=5)
        ttk.Label(gen_frame, text="Phase Shift:").grid(row=0, column=2, sticky="w")
        self.phase_entry = ttk.Entry(gen_frame, width=10)
        self.phase_entry.grid(row=0, column=3, padx=5)
        ttk.Label(gen_frame, text="Analog Freq (Hz):").grid(row=1, column=0, sticky="w")
        self.analog_freq_entry = ttk.Entry(gen_frame, width=10)
        self.analog_freq_entry.grid(row=1, column=1, padx=5)
        ttk.Label(gen_frame, text="Sampling Freq (Hz):").grid(row=1, column=2, sticky="w")
        self.sampling_freq_entry = ttk.Entry(gen_frame, width=10)
        self.sampling_freq_entry.grid(row=1, column=3, padx=5)
        btn_frame = ttk.Frame(gen_frame)
        btn_frame.grid(row=2, column=0, columnspan=4, pady=5)
        ttk.Button(btn_frame, text="Generate Sine", command=lambda: self._generate_signal('sine')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Generate Cosine", command=lambda: self._generate_signal('cosine')).pack(side=tk.LEFT, padx=5)


        # قسم 2: عرض الإشارات والعمليات (Task 1)
        op_frame = ttk.Labelframe(self.scrollable_frame, text="2. Available Signals & Operations (Task 1)")
        op_frame.pack(fill="x", padx=10, pady=5)
        # (باقي المحتويات)
        ttk.Label(op_frame, text="Select Signal(s): (use Ctrl+Click for multiple)").pack()
        self.signal_listbox = tk.Listbox(op_frame, height=8, exportselection=False, selectmode=tk.MULTIPLE)
        self.signal_listbox.pack(fill="x", padx=5, pady=5)
        plot_btn_frame = ttk.Frame(op_frame)
        plot_btn_frame.pack(fill="x")
        ttk.Button(plot_btn_frame, text="Plot Selected", command=self._plot_selected).pack(side=tk.LEFT, padx=5)
        self.plot_mode = tk.StringVar(value="discrete")
        ttk.Radiobutton(plot_btn_frame, text="Discrete", variable=self.plot_mode, value="discrete").pack(side=tk.LEFT)
        ttk.Radiobutton(plot_btn_frame, text="Continuous", variable=self.plot_mode, value="continuous").pack(side=tk.LEFT)
        op_btn_frame = ttk.Frame(op_frame)
        op_btn_frame.pack(fill="x", pady=5)
        ttk.Button(op_btn_frame, text="Add", command=lambda: self._perform_operation('add')).pack(side=tk.LEFT, padx=2)
        ttk.Button(op_btn_frame, text="Subtract", command=lambda: self._perform_operation('subtract')).pack(side=tk.LEFT, padx=2)
        ttk.Button(op_btn_frame, text="Fold", command=lambda: self._perform_operation('fold')).pack(side=tk.LEFT, padx=2)
        shift_frame = ttk.Frame(op_frame)
        shift_frame.pack(fill="x", pady=5)
        ttk.Button(shift_frame, text="Shift", command=lambda: self._perform_operation('shift')).pack(side=tk.LEFT, padx=2)
        self.shift_entry = ttk.Entry(shift_frame, width=5)
        self.shift_entry.pack(side=tk.LEFT)
        mult_frame = ttk.Frame(op_frame)
        mult_frame.pack(fill="x", pady=5)
        ttk.Button(mult_frame, text="Multiply", command=lambda: self._perform_operation('multiply')).pack(side=tk.LEFT, padx=2)
        self.mult_entry = ttk.Entry(mult_frame, width=5)
        self.mult_entry.pack(side=tk.LEFT)
        
        # قسم 3: Quantization (Task 3)
        self.quant_frame = QuantizationFrame(self.scrollable_frame, self.loaded_signals)
        self.quant_frame.pack(fill="x", expand=True, padx=10, pady=5)
        
                #  (Task 5) 
        task5_frame = ttk.Labelframe(self.scrollable_frame, text="4. Task 5: Filters & Convolution")
        task5_frame.pack(fill="x", padx=10, pady=5)

        # --- Moving Average ---
        mv_frame = ttk.Frame(task5_frame)
        mv_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(mv_frame, text="Moving Average (Select 1)", command=self._handle_moving_average).pack(side=tk.LEFT, padx=2)
        ttk.Label(mv_frame, text="Window Size:").pack(side=tk.LEFT)
        self.mv_window_entry = ttk.Entry(mv_frame, width=5)
        self.mv_window_entry.insert(0, "3") # قيمة افتراضية
        self.mv_window_entry.pack(side=tk.LEFT)

        # --- Derivatives ---
        deriv_frame = ttk.Frame(task5_frame)
        deriv_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(deriv_frame, text="First Derivative (Select 1)", command=self._handle_first_derivative).pack(side=tk.LEFT, padx=2)
        ttk.Button(deriv_frame, text="Second Derivative (Select 1)", command=self._handle_second_derivative).pack(side=tk.LEFT, padx=2)

        # --- Convolution ---
        conv_frame = ttk.Frame(task5_frame)
        conv_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(conv_frame, text="Convolution (Select 2)", command=self._handle_convolution).pack(side=tk.LEFT, padx=2)
        
        # Task  (Fourier Transform) 
        task3_frame = ttk.Labelframe(self.scrollable_frame, text="3. Task : Fourier Transform")
        task3_frame.pack(fill="x", padx=10, pady=5)

        # إدخال تردد العينات
        freq_input_frame = ttk.Frame(task3_frame)
        freq_input_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(freq_input_frame, text="Sampling Frequency (Fs):").pack(side=tk.LEFT)
        self.fs_entry = ttk.Entry(freq_input_frame, width=10)
        self.fs_entry.insert(0, "1") 
        self.fs_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(freq_input_frame, text="Hz").pack(side=tk.LEFT)

        # أزرار التنفيذ
        dft_btn_frame = ttk.Frame(task3_frame)
        dft_btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(dft_btn_frame, text="Apply DFT (Freq & Phase)", command=self._handle_dft).pack(side=tk.LEFT, padx=5)
        ttk.Button(dft_btn_frame, text="Apply IDFT (Reconstruct)", command=self._handle_idft).pack(side=tk.LEFT, padx=5)
        
        #  Task 6 (Correlation) 
        task6_frame = ttk.Labelframe(self.scrollable_frame, text=" Correlation & Time Delay")
        task6_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(task6_frame, text=" Correlation (Select 2)", command=self._handle_correlation).pack(side=tk.LEFT, padx=5, pady=5)

        # جزء Time Delay
        td_frame = ttk.Frame(task6_frame)
        td_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(td_frame, text="Fs (Hz):").pack(side=tk.LEFT)
        self.fs_task6_entry = ttk.Entry(td_frame, width=8)
        self.fs_task6_entry.insert(0, "100") # قيمة افتراضية
        self.fs_task6_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(td_frame, text="Calculate Time Delay (Select 2)", command=self._handle_time_delay).pack(side=tk.LEFT, padx=5)
        # ... (بعد كود Task 6 Correlation & Time Delay) ...

        # =================================================================
        #  --- (2) بداية الإضافة: Task B: FIR Filters ---
        # =================================================================
        fir_frame = ttk.Labelframe(self.scrollable_frame, text="Task B: FIR Filters")
        fir_frame.pack(fill="x", padx=10, pady=5)
        # اختيار طريقة التنفيذ
        method_frame = ttk.Frame(fir_frame)
        method_frame.pack(fill="x", padx=5, pady=5)
        self.fir_method_var =  tk.StringVar(value="Direct")
        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        
        self.conv_method_var = tk.StringVar(value="Direct")
        ttk.Radiobutton(method_frame, text="Direct Convolution", variable=self.conv_method_var, value="Direct").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(method_frame, text="Fast Convolution (FFT)", variable=self.conv_method_var, value="Fast").pack(side=tk.LEFT, padx=5)


        # 1. اختيار النوع
        type_frame = ttk.Frame(fir_frame)
        type_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(type_frame, text="Filter Type:").pack(side=tk.LEFT)
        self.combo_fir_type = ttk.Combobox(type_frame, values=["Low pass", "High pass", "Band pass", "Band stop"], state="readonly", width=15)
        self.combo_fir_type.current(0)
        self.combo_fir_type.pack(side=tk.LEFT, padx=5)
        # ربط حدث التغيير لتفعيل/تعطيل F2
        self.combo_fir_type.bind("<<ComboboxSelected>>", self._update_fir_inputs_visibility)

        # 2. المدخلات (Parameters)
        params_frame = ttk.Frame(fir_frame)
        params_frame.pack(fill="x", padx=5, pady=2)
        
        # الصف الأول: FS & Stop Attenuation
        ttk.Label(params_frame, text="FS:").grid(row=0, column=0, sticky="e")
        self.txt_fir_fs = ttk.Entry(params_frame, width=8)
        self.txt_fir_fs.grid(row=0, column=1, padx=2, pady=2)
        
        ttk.Label(params_frame, text="Stop Atten (dB):").grid(row=0, column=2, sticky="e")
        self.txt_fir_stop = ttk.Entry(params_frame, width=8)
        self.txt_fir_stop.grid(row=0, column=3, padx=2, pady=2)

        # الصف الثاني: F1 (Cutoff) & F2
        ttk.Label(params_frame, text="F1 (Cutoff):").grid(row=1, column=0, sticky="e")
        self.txt_fir_f1 = ttk.Entry(params_frame, width=8)
        self.txt_fir_f1.grid(row=1, column=1, padx=2, pady=2)

        ttk.Label(params_frame, text="F2 (Band only):").grid(row=1, column=2, sticky="e")
        self.txt_fir_f2 = ttk.Entry(params_frame, width=8, state="disabled") # Disabled by default
        self.txt_fir_f2.grid(row=1, column=3, padx=2, pady=2)

        # الصف الثالث: Transition Band
        ttk.Label(params_frame, text="Trans. Band:").grid(row=2, column=0, sticky="e")
        self.txt_fir_trans = ttk.Entry(params_frame, width=8)
        self.txt_fir_trans.grid(row=2, column=1, padx=2, pady=2)

        # 3. زر التنفيذ
        action_frame = ttk.Frame(fir_frame)
        action_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(action_frame, text="Design, Apply & Save", command=self._handle_fir_filter).pack(fill="x")
                # 4. زر المقارنة (Compare)

        ttk.Button(fir_frame, text="Compare with Expected Output", command=self._handle_fir_comparison).pack(fill="x", pady=2)

    


    # --- دوال التحكم (Handlers) ---

    def _update_all_lists(self):
        self.signal_listbox.delete(0, tk.END)
        names = list(self.loaded_signals.keys())
        for name in names:
            self.signal_listbox.insert(tk.END, name)
        if hasattr(self, 'quant_frame'):
            self.quant_frame.update_signal_list()

    def _load_signal(self):
        filepath = filedialog.askopenfilename(
            initialdir="TestCases",
            title="Select a Signal File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not filepath: return
        try:
            indices, values = read_signal(filepath) 
            
            if values is None or len(values) == 0:
                messagebox.showerror("Error", "Could not read valid samples from the file.")
                return
            filename = os.path.basename(filepath)
            self.loaded_signals[filename] = (indices, values)
            self._update_all_lists()
            print(f"Loaded '{filename}' successfully with {len(values)} samples.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or process file: {e}")

    def _generate_signal(self, wave_type):
        try:
            amp = float(self.amp_entry.get())
            phase = float(self.phase_entry.get())
            a_freq = float(self.analog_freq_entry.get())
            s_freq = float(self.sampling_freq_entry.get())
        except ValueError:
            messagebox.showerror("Error", "All generation inputs must be valid numbers.")
            return

        indices, values = dsp.generate_sinusoidal_signal(wave_type, amp, phase, a_freq, s_freq)
        
        if values is None:
            messagebox.showerror("Error", "Could not generate signal.")
            return
            
        self.generated_signal_count += 1
        name = f"Gen_{wave_type}_{self.generated_signal_count}"
        self.loaded_signals[name] = (indices, values)
        self._update_all_lists()

    def _plot_selected(self):
        selected_indices = self.signal_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one signal to plot.")
            return
        self.fig.clear()    
        self.ax = self.fig.add_subplot(111)
        plot_mode = self.plot_mode.get()
        
        for i in selected_indices:
            name = self.signal_listbox.get(i)
            indices, values = self.loaded_signals[name]
            
            if plot_mode == 'discrete':
                self.ax.stem(indices, values, label=name)
            else: 
                self.ax.plot(indices, values, label=name)
        
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlabel("Index (n)")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def _get_selected_signals_for_ops(self, num_required):
        selected_indices = self.signal_listbox.curselection()
        if len(selected_indices) != num_required:
            messagebox.showerror("Error", f"Please select exactly {num_required} signal(s) for this operation.")
            return None
        
        signals = []
        for i in selected_indices:
            name = self.signal_listbox.get(i)
            signals.append(self.loaded_signals[name])
        return signals

    def _perform_operation(self, op_type):
        # (هذه الدالة خاصة بـ Task 1)
        new_indices, new_values = None, None
        new_name = ""

        try:
            if op_type == 'add' or op_type == 'subtract':
                signals = self._get_selected_signals_for_ops(2)
                if not signals: return
                
                sig1_name = self.signal_listbox.get(self.signal_listbox.curselection()[0])
                sig2_name = self.signal_listbox.get(self.signal_listbox.curselection()[1])

                if op_type == 'add':
                    new_indices, new_values = dsp.add_signals(signals[0], signals[1])
                    new_name = f"Add({sig1_name}, {sig2_name})"
                else:
                    new_indices, new_values = dsp.subtract_signals(signals[0], signals[1])
                    new_name = f"Sub({sig1_name}, {sig2_name})"

            elif op_type == 'fold' or op_type == 'multiply' or op_type == 'shift':
                signals = self._get_selected_signals_for_ops(1)
                if not signals: return
                sig_name = self.signal_listbox.get(self.signal_listbox.curselection()[0])

                if op_type == 'fold':
                    new_indices, new_values = dsp.fold_signal(signals[0])
                    new_name = f"Fold({sig_name})"
                
                elif op_type == 'multiply':
                    k = float(self.mult_entry.get())
                    new_indices, new_values = dsp.multiply_by_constant(signals[0], k)
                    new_name = f"Mult({sig_name}, {k})"
                
                elif op_type == 'shift':
                    k = int(self.shift_entry.get())
                    new_indices, new_values = dsp.shift_signal(signals[0], k)
                    new_name = f"Shift({sig_name}, {k})"

            if new_indices is not None:
                self.loaded_signals[new_name] = (new_indices, new_values)
                self._update_all_lists()

        except ValueError:
            messagebox.showerror("Error", "Invalid constant for Multiply/Shift. Must be a number.")
        except Exception as e:
            messagebox.showerror("Error", f"Operation (Task 1) failed: {e}")

    # --- !! دوال التحكم  الخاصة بـ Task 5 !! ---

    def _handle_moving_average(self):
        try:
            signals = self._get_selected_signals_for_ops(1)
            if not signals: return
            
            M = int(self.mv_window_entry.get())

            sig_name = self.signal_listbox.get(self.signal_listbox.curselection()[0])
            indices, values = signals[0]
            
            new_indices, new_values = moving_average(indices, values, M)
            new_name = f"MovAvg({sig_name}, M={M})"
            
            self.loaded_signals[new_name] = (new_indices, new_values)
            self._update_all_lists()

        except ValueError:
            messagebox.showerror("Error", "Window size must be a valid integer.")
        except Exception as e:
            messagebox.showerror("Error", f"Moving Average failed: {e}")

    def _handle_first_derivative(self):
        try:
            signals = self._get_selected_signals_for_ops(1)
            if not signals: return
            
            sig_name = self.signal_listbox.get(self.signal_listbox.curselection()[0])
            indices, values = signals[0]

            new_indices, new_values = first_derivative(indices, values)
            new_name = f"1stDeriv({sig_name})"
            
            self.loaded_signals[new_name] = (new_indices, new_values)
            self._update_all_lists()

        except Exception as e:
            messagebox.showerror("Error", f"First Derivative failed: {e}")

    def _handle_second_derivative(self):
        try:
            signals = self._get_selected_signals_for_ops(1)
            if not signals: return
            
            sig_name = self.signal_listbox.get(self.signal_listbox.curselection()[0])
            indices, values = signals[0]

            new_indices, new_values = second_derivative(indices, values)
            new_name = f"2ndDeriv({sig_name})"
            
            self.loaded_signals[new_name] = (new_indices, new_values)
            self._update_all_lists()

        except Exception as e:
            messagebox.showerror("Error", f"Second Derivative failed: {e}")

    def _handle_convolution(self):
        try:
            signals = self._get_selected_signals_for_ops(2)
            if not signals: return
            
            sig1_name = self.signal_listbox.get(self.signal_listbox.curselection()[0])
            sig2_name = self.signal_listbox.get(self.signal_listbox.curselection()[1])
            (indices1, values1) = signals[0]
            (indices2, values2) = signals[1]

            new_indices, new_values = convolution(indices1, values1, indices2, values2)
            new_name = f"Conv({sig1_name}, {sig2_name})"
            
            self.loaded_signals[new_name] = (new_indices, new_values)
            self._update_all_lists()

        except Exception as e:
            messagebox.showerror("Error", f"Convolution failed: {e}")
    
    # --- !! دوال التحكم الخاصة بـ Task fourier !! ---

    def _handle_dft(self):
        try:
            signals = self._get_selected_signals_for_ops(1)
            if not signals: return
            
            indices, values = signals[0]
            
            try:
                fs = float(self.fs_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid Sampling Frequency (Fs).")
                return

            freqs, amps, phases, complex_result = tf.discrete_fourier_transform(values, fs, is_inverse=False)
            
            self.last_dft_complex_result = complex_result

            self.fig.clear() 
            
            ax1 = self.fig.add_subplot(211) 
            ax1.stem(freqs, amps)
            ax1.set_title("Frequency Domain: Amplitude")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True)

            ax2 = self.fig.add_subplot(212) 
            ax2.stem(freqs, phases)
            ax2.set_title("Frequency Domain: Phase")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Phase (radians)")
            ax2.grid(True)

            self.fig.tight_layout() 
            self.canvas.draw()
            
            print("DFT Applied successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"DFT failed: {e}")

    def _handle_idft(self):
        try:
            if self.last_dft_complex_result is None:
                messagebox.showwarning("Warning", "Please apply DFT first to have data for IDFT.")
                return

            fs = float(self.fs_entry.get())

            indices, reconstructed_values = tf.discrete_fourier_transform(self.last_dft_complex_result, fs, is_inverse=True)
            
            
            new_name = f"Reconstructed_Signal_{self.generated_signal_count}"
            self.generated_signal_count += 1
            self.loaded_signals[new_name] = (indices, reconstructed_values)
            self._update_all_lists()
            
            self.fig.clear()
            self.ax = self.fig.add_subplot(111) 
            self.ax.stem(indices, reconstructed_values, label=new_name)
            self.ax.set_title("Reconstructed Signal (IDFT)")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()
            
            print("IDFT Applied successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"IDFT failed: {e}")  
     
    # --- !! دوال Task 6 (Correlation) !! ---

    def _handle_correlation(self):
        try:
            signals = self._get_selected_signals_for_ops(2)
            if not signals: return
            
            
            sig1_name = self.signal_listbox.get(self.signal_listbox.curselection()[0])
            sig2_name = self.signal_listbox.get(self.signal_listbox.curselection()[1])
            
            ind1, val1 = signals[0]
            ind2, val2 = signals[1]

            corr_ind, corr_val = corr.fast_correlation(ind1, val1, ind2, val2)
            
            new_name = f"Corr({sig1_name}, {sig2_name})"
            self.loaded_signals[new_name] = (corr_ind, corr_val)
            self._update_all_lists()
            
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.ax.stem(corr_ind, corr_val, label=new_name)
            self.ax.set_title(f"Cross Correlation: {sig1_name} & {sig2_name}")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Correlation failed: {e}")

    def _handle_time_delay(self):
        try:
            signals = self._get_selected_signals_for_ops(2)
            if not signals: return
            
            ind1, val1 = signals[0]
            ind2, val2 = signals[1]
            
            try:
                fs = float(self.fs_task6_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid Sampling Frequency (Fs).")
                return

            lag, delay = corr.fast_time_delay(ind1, val1, ind2, val2, fs)
            
            messagebox.showinfo("Time Delay Result", f"Lag Index (j): {lag}\nTime Delay: {delay} seconds")
            
        except Exception as e:
    
           messagebox.showerror("Error", f"Time Delay calc failed: {e}") 
    # =================================================================
    #  --- FIR Filter ---
    # =================================================================

    def _update_fir_inputs_visibility(self, event=None):
        """تفعيل أو تعطيل خانة F2 حسب نوع الفلتر"""
        ftype = self.combo_fir_type.get()
        if "Band" in ftype:
            self.txt_fir_f2.configure(state="normal")
        else:
            self.txt_fir_f2.delete(0, tk.END)
            self.txt_fir_f2.configure(state="disabled")

    def _handle_fir_filter(self):
        try:
            # 1. التأكد من اختيار إشارة
            sigs = self._get_selected_signals_for_ops(1)
            if not sigs: return
            s_name = self.signal_listbox.get(self.signal_listbox.curselection()[0])
            indices, values = sigs[0]

            # 2. قراءة المدخلات
            ftype = self.combo_fir_type.get()
            fs = float(self.txt_fir_fs.get())
            stop_att = float(self.txt_fir_stop.get())
            trans = float(self.txt_fir_trans.get())
            f1 = float(self.txt_fir_f1.get())
            f2 = 0
            if "Band" in ftype: f2 = float(self.txt_fir_f2.get())

            # 3. تصميم الفلتر وحفظ المعاملات
            coeffs = fir.design_fir_coefficients(ftype, fs, stop_att, trans, f1, f2)
            fir.save_coefficients_to_file("coefficients.txt", coeffs)
            
            # 4. تطبيق الفلتر
            method = self.fir_method_var.get()
            if method == "Direct":
                filtered_vals = fir.apply_filter_direct(values, coeffs)
            else:
                filtered_vals = fir.apply_filter_fast(values, coeffs)
            
            # 5. [هام جداً] تخزين النتيجة للمقارنة
            self.filtered_data = filtered_vals 
            
            # 6. إضافة الإشارة للقائمة وتحديث الرسم
            # (بنحسب بداية الرسم عشان يبان الشفت الزمني صح)
            start_idx = -int((len(coeffs) - 1) / 2)
            new_ind = np.arange(start_idx, start_idx + len(filtered_vals))
            
            out_filename = f"ecg_{ftype.replace('','_')}_filtered_output.txt"
            fir.save_signal_to_file(out_filename, new_ind, filtered_vals)
            new_name = f"Filt_{s_name}_{ftype}"
            self.loaded_signals[new_name] = (new_ind, filtered_vals)
            self._update_all_lists()

            # 7. الرسم
            self.fig.clear()
            
            ax1 = self.fig.add_subplot(211)
            ax1.plot(indices, values, label="Original", alpha=0.6)
            ax1.plot(new_ind, filtered_vals, label="Filtered", color='red')
            ax1.set_title(f"Filtering Result ({method})")
            ax1.legend()
            ax1.grid(True)

            ax2 = self.fig.add_subplot(212)
            ax2.stem(coeffs)
            ax2.set_title("Filter Coefficients")
            ax2.grid(True)

            self.fig.tight_layout()
            self.canvas.draw()
            
            messagebox.showinfo("Success", f"Filter Designed (N={len(coeffs)}).\nSignal Filtered & Coefficients saved.")

        except ValueError: messagebox.showerror("Error", "Check inputs (Ensure numeric values).")
        except Exception as e: messagebox.showerror("Error", str(e))

    def _handle_fir_comparison(self):
        # التأكد من وجود بيانات للمقارنة
        if self.filtered_data is None:
            messagebox.showwarning("Warning", "Please apply the filter first (Design & Apply)!")
            return
        
        filepath = filedialog.askopenfilename(initialdir="TestCases", title="Select Expected Output", filetypes=[("Text Files", "*.txt")])
        if not filepath: return
        
        # استدعاء دالة المقارنة من ملف fir
        res = fir.compare_with_file(filepath, self.filtered_data)
        
        if "Passed" in res: messagebox.showinfo("Result", res)
        else: messagebox.showerror("Result", res)            

if __name__ == "__main__":
    app = SignalProcessingApp()
    app.mainloop()