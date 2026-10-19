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
