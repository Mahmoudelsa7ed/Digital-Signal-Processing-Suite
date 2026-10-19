# 📊 Digital Signal Processing (DSP) Framework & GUI

A comprehensive **Python** desktop application and algorithmic toolkit built to process, analyze, filter, and classify digital signals. This project implements core Digital Signal Processing algorithms from scratch alongside automated testing scripts and an interactive Graphical User Interface (GUI).

---

## 🚀 Key Features & Implemented Modules

* **Signal Operations (`dsp_operations.py`, `dsp_functions.py`):**
  * Generation and manipulation of discrete signals.
  * Arithmetic operations: addition, subtraction, multiplication, squaring, shifting, folding, normalization, and accumulation.
* **Signal Quantization (`task3_quantization.py`, `task3_gui.py`):**
  * Quantizes continuous-amplitude signals based on specified levels or number of bits, computing quantized values, encoded binary representations, and quantization error.
* **Frequency Domain & Fourier Transform (`Task_fourier.py`):**
  * Implementation of Discrete Fourier Transform (**DFT**), Inverse Discrete Fourier Transform (**IDFT**), Fast Fourier Transform (**FFT**), and DC component removal.
* **FIR Filter Design & Convolution (`FIR_Filter.py`):**
  * Designs Finite Impulse Response (**FIR**) filters (Low-Pass, High-Pass, Band-Pass, and Band-Stop) using various windowing functions (Rectangular, Hanning, Hamming, Blackman).
  * Fast and direct signal filtering and resampling.
* **Correlation & Time Delay Analysis (`correlation.py`):**
  * Computes cross-correlation and normalized auto-correlation between signals (direct and frequency-domain approaches).
* **Signal Classification (`Classification.py`):**
  * Template-matching classifier utilizing correlation coefficients to categorize test signals into distinct classes (`Class1` vs. `Class2`).
* **Unified Main Application (`main_app.py`):**
  * Central GUI integrating all tasks and DSP modules into a single interactive workspace.

---

## 📂 Project Structure

* `main_app.py`: Primary GUI application entry point.
* `dsp_operations.py` & `dsp_functions.py`: Core signal arithmetic and helper functions.
* `task3_quantization.py` & `task3_gui.py`: Signal quantization logic and dedicated interface.
* `Task_fourier.py`: Frequency domain analysis (DFT/IDFT).
* `FIR_Filter.py`: FIR filter design, windowing, and convolution.
* `correlation.py` & `Classification.py`: Signal correlation and pattern classification.
* `Compare.py`, `run_tests.py`, `QuanTest1.py`, `QuanTest2.py`: Automated evaluation scripts comparing output against reference test cases (`Testcase 1–8`, `TestCases`, `Class1`, `Class2`).

---
## 🖥️ User Guide: How to Use the Application (`main_app.py`)

Once you launch the application using `python main_app.py`, the interactive GUI will open. You can test the different DSP modules using the provided `.txt` signal files located in the `TestCases` and `Testcase 1–8` folders:

### 1. Loading & Generating Signals
* **Load Signal from File:** Click the **Load / Browse Signal** button and select any `.txt` signal file (e.g., `Test1.txt`, `Test2.txt`, or files inside the `TestCases` folders). The signal will be parsed and plotted automatically.
* **Signal Generation:** Alternatively, you can generate a custom sinusoidal signal (`Sine` or `Cosine`) by entering the **Amplitude (A)**, **Analog Frequency (F)**, **Sampling Frequency (Fs)**, and **Phase Shift**.

### 2. Time-Domain Operations
* Load one or two signals to perform arithmetic and manipulation operations:
  * **Two-Signal Operations:** Addition, Subtraction, and Cross-Correlation.
  * **Single-Signal Operations:** Multiplication by a constant, Squaring, Shifting (advance/delay), Folding (time-reversal), Normalization (`[0, 1]` or `[-1, 1]`), and Accumulation.

### 3. Signal Quantization
* Navigate to the **Quantization** section (or run `task3_gui.py`).
* Load your input signal file, choose whether to quantize by **Number of Bits** or **Number of Levels**, and enter the value.
* The app will display the **Interval Index**, **Quantized Signal**, **Encoded Binary Values**, and **Quantization Error**, and compare them against the reference test outputs (`QuanTest1.py` / `QuanTest2.py`).

### 4. Frequency Domain (DFT / IDFT / DCT)
* Load a time-domain signal and enter the **Sampling Frequency (Fs)**.
* Apply **DFT (Discrete Fourier Transform)** to view the **Frequency vs. Amplitude** and **Frequency vs. Phase** spectra.
* Modify amplitudes/phases or remove the **DC Component**, or load a polar frequency-domain file to reconstruct the original time-domain signal using **IDFT**.

### 5. FIR Filtering & Resampling
* Select the **FIR Filter** module:
  * Choose the **Filter Type** (`Low-Pass`, `High-Pass`, `Band-Pass`, or `Band-Stop`).
  * Enter the filter specifications: **Sampling Frequency (Fs)**, **Stopband Attenuation**, **Transition Band**, and **Cutoff Frequency (Fc for Low/High-Pass, or F1 and F2 for Band-Pass/Stop)**.
  * The application automatically selects the appropriate window function (**Rectangular, Hanning, Hamming, or Blackman**), computes the filter coefficients, and applies the filter to your loaded signal (via Direct or Fast Convolution).
  * You can also export the generated filter coefficients or filtered signal to a `.txt` file.

### 6. Correlation & Signal Classification
* **Correlation:** Load two signals from the correlation test folders to compute normalized cross-correlation and estimate time delay.
* **Classification:** Load an unknown test signal from the `TestCases` folder; the system correlates it against the training templates in `Class1` and `Class2` and displays the predicted class.
