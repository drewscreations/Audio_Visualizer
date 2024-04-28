import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from scipy.signal import spectrogram
import numpy as np
import socket
import threading
import time
import pickle


class RingBuffer:
    """A simple ring buffer to manage real-time data streams, now suitable for 2D arrays."""

    def __init__(self, size=1024, max_length=2):
        """
        
        Initialize the buffer with a fixed size and maximum length for 2D storage.
        :param size: The maximum size of each array in the buffer.
        :param max_length: Maximum number of arrays to store.
        """
        self.size = size
        if self.size == 1:
            self.data = []*max_length#np.zeros(max_length)
        else:
            self.data = [np.zeros(size)] * max_length
        
        self.max_length = max_length
        self.index = 0

    def add(self, x):
        """
        Add a 2D array to the buffer.
        :param x: New 2D array to add to the buffer.
        """
        if len(self.data) >= self.max_length:
            self.data.pop(0)  # Remove oldest data if at capacity
        self.data.append(x)

    def get_all(self):
        """
        Get all the 2D arrays in the buffer concatenated along time axis.
        """
        return np.hstack(self.data) if self.data else np.array([])
    
    def get_last(self, last):
        return np.hstack(self.data)[-5:] if self.data else np.array([])

class SignalGenerator:
    """Handles the generation and manipulation of signals."""

    def __init__(self, length=1000, frequency=0.2, buffer_size=10):
        """
        Initialize default parameters for the signal.
        :param length: Number of data points in the signal.
        :param frequency: Frequency of the base signal.
        """
        self.length = length
        self.frequency = frequency
        self.buffer = RingBuffer(size=1, max_length=buffer_size)

    def generate_signal(self):
        """Generate a simple sinusoidal signal based on the initialized frequency and length."""
        t = np.linspace(0, self.frequency * 2 * np.pi, self.length)
        return np.sin(t)

    def add_noise(self, signal, amplitude=1, noise_frequency=1, duration=1):
        """
        Add noise to the signal.
        :param signal: The original signal.
        :param amplitude: Amplitude of the noise.
        :param noise_frequency: Frequency of the noise.
        :param duration: Duration over which noise is active as a percentage of total signal length.
        """
        noise = amplitude * np.random.normal(size=int(self.length * (duration)))
        full_noise = np.zeros_like(signal)
        full_noise[:len(noise)] = noise  # Apply noise at the start of the signal
        return signal + full_noise
    
    def get_signal(self):
        while True:
            signal = self.generate_signal()
            amp = np.random.randint(0, 10)/10
            freq = np.random.randint(0, 10)/10
            dur = np.random.randint(0, 10)/10
            signal = self.add_noise(signal, amp, freq, dur)
            self.buffer.add(signal)
            time.sleep(0.1)
            #return signal
        

class SocketSignal:
    """Handles the generation and reception of signals, now with socket streaming."""

    def __init__(self, host='localhost', port=6000, buffer_size=1024):
        """
        Initialize the socket connection and buffer.
        :param host: Host address.
        :param port: Port number.
        :param buffer_size: Size of the ring buffer.
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.buffer = RingBuffer(size=1, max_length=buffer_size)

    def stream_data(self):
        """
        Continuously read data from the socket and add to the buffer.
        """
        while True:
            data = self.sock.recv(1024)
            numbers = np.frombuffer(data, dtype=np.float32)  # Assuming float data
            for number in numbers:
                self.buffer.add(number)

class SpectrogramCalculator:
    """Calculates and buffers spectrograms, with updates via callback."""

    def __init__(self, callback, max_threads=5):
        self.callback = callback
        self.buffer = RingBuffer(size=(129, 100), max_length=10)  # Adjust dimensions as needed
        self.max_threads = max_threads
        self.active_threads = []

    def calculate_spectrogram(self, signal):
        """
        Manage threads to start spectrogram calculations.
        :param signal: The signal to process.
        """
        # Clean up completed threads
        self.active_threads = [t for t in self.active_threads if t.is_alive()]
        
        # Check if we need to remove the oldest thread
        if len(self.active_threads) >= self.max_threads:
            oldest_thread = self.active_threads.pop(0)  # Remove the oldest active thread
            if oldest_thread.is_alive():
                oldest_thread.join()  # Ensure the thread is completed before removing

        thread = threading.Thread(target=self._run_calculation, args=(signal,))
        thread.daemon = True
        thread.start()

    def _run_calculation(self, signal):
        frequencies, times, Sxx = spectrogram(signal, fs=8000)
        self.buffer.add(Sxx)  # Add new spectrogram to the buffer
        self.callback(frequencies, self.buffer.get_all())  # Pass concatenated data


class SignalApp:
    """Application class with synchronized time window updates for plots."""

    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Signal Analysis App")

        self.signal_generator = SignalGenerator()
        threading.Thread(target=self.signal_generator.get_signal, daemon=True).start()

        self.spectrogram_calculator = SpectrogramCalculator(self.update_spectrogram_plot)
        
        self.setup_plots()
        self.setup_controls()

    def setup_plots(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.ax1.set_title('Real-Time Signal with Noise')
        self.ax2.set_title('Stitched Real-Time Spectrogram')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=6, sticky='nsew')
        # Initialize a placeholder for the spectrogram plot
        self.pcm = self.ax2.pcolormesh(np.zeros((129, 100)), norm=Normalize(vmin=-40, vmax=0))  # Adjust size and limits as needed
        
        
        # Animation
        self.update_interval = 100 #ms
        self.animation_save_ct = 100
        self.anim = animation.FuncAnimation(self.fig, self.update_plots, interval=self.update_interval, save_count=self.animation_save_ct, blit=False)

    def update_plots(self, frame):
        signal = np.array(self.signal_generator.buffer.get_all())
        self.ax1.clear()
        self.ax1.plot(signal)

        # Trigger a new calculation each update
        self.spectrogram_calculator.calculate_spectrogram(signal)
        return self.ax1, self.ax2

    def update_spectrogram_plot(self, frequencies, Sxx_stitched):
        """Update the spectrogram plot with stitched data."""
        self.ax2.clear()
        max_size = 1028
        #print(shape(Sxx_stitched))
        #Sxx_stitched = Sxx_stitched[:-max_size]
        self.ax2.pcolormesh(Sxx_stitched, cmap='viridis')

        self.canvas.draw_idle()

    def setup_controls(self):
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=0, sticky='ew')
        control_frame.grid_columnconfigure(0, weight=1)

        self.window_size_slider = tk.Scale(control_frame, from_=100, to_=1000, label="Window Size", orient=tk.HORIZONTAL)
        self.window_size_slider.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        self.window_size_slider.bind("<Motion>", self.adjust_window_size)

    def adjust_window_size(self, motion):
        #print(motion)
        pass

def main():
    """Run the main application."""
    root = tk.Tk()
    app = SignalApp(root)
    app.root.mainloop()

if __name__ == '__main__':
    main()
