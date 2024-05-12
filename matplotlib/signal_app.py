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

class RingBuffer1D:
    """A simple ring buffer to manage real-time data streams."""
    
    def __init__(self, size=160000):  # 8000 samples/second * 20 seconds
        self.data = np.zeros(size, dtype=np.float32)
        self.size = size
        self.index = 0

    def add(self, x):
        n = len(x)
        if self.index + n < self.size:
            self.data[self.index:self.index + n] = x
        else:
            end_size = self.size - self.index
            self.data[self.index:] = x[:end_size]
            self.data[:n - end_size] = x[end_size:]
        self.index = (self.index + n) % self.size

    def get_all(self):
        return self.data

    def get_last_seconds(self, seconds):
        """Get the last 'seconds' worth of data."""
        samples = seconds * 8000
        return self.data[self.index - samples:self.index] if self.index >= samples else self.data[-samples:]


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
    
    def get_last_seconds(self, seconds):
        """Get the last 'seconds' worth of data."""
        samples = seconds * 8000
        return np.hstack(self.data[self.index - samples:self.index]) if self.index >= samples else np.array([])#np.hstack(self.data[-samples:])


class SignalGenerator:
    """Handles the generation and manipulation of signals."""

    def __init__(self, ready_cb, length=800, frequency=0.2, buffer_size=10):
        """
        Initialize default parameters for the signal.
        :param length: Number of data points in the signal.
        :param frequency: Frequency of the base signal.
        """
        self.length = length
        self.frequency = frequency
        self.ready_cb = ready_cb
        #self.buffer = RingBuffer(size=1, max_length=buffer_size)
        self.fs = 8000 #hz
        self.buffer_duration_s = 20
        self.buffer = RingBuffer1D(size=self.fs*self.buffer_duration_s)
        self.signal_block_duration_s = 1
        self.signal_block_overlap_s = 0.5
        self.last_signal_block = np.array([])

    def generate_signal(self, freq = None):
        """Generate a simple sinusoidal signal based on the initialized frequency and length."""
        if freq == None:
            freq = self.frequency
        t = np.linspace(0, freq * 2 * np.pi, self.length)
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
    
    def add_to_signal(self):
        while True:
            signal = self.generate_signal(40)
            if np.random.randint(0, 10) <= 5:
                pass
            else:
                signal = self.generate_signal(80)
                amp = np.random.randint(0, 10)/10
                freq = np.random.randint(0, 10)/10
                dur = np.random.randint(0, 10)/10
                signal = self.add_noise(signal, amp, freq, dur)
                self.buffer.add(signal)
                #if len(self.last_signal_block)>self.signal_block_duration_s*self.fs:
                #print(np.shape(self.last_signal_block))
                if np.shape(self.last_signal_block)[0]<self.signal_block_duration_s*self.fs:
                    self.last_signal_block = np.append(self.last_signal_block, signal)
                else:
                    self.ready_cb(self.last_signal_block)
                    self.last_signal_block = np.array([])
                    
            sleep_dur = self.length/self.fs
            time.sleep(round(sleep_dur, 3))
            #return signal
            
    def get_signal(self, seconds):
        signal = self.buffer.get_last_seconds(seconds)
        return signal
        
        
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

    def __init__(self, max_threads=5):
        self.buffer = RingBuffer(size=(20001, 47), max_length=10)  # Adjust dimensions as needed
        #self.buffer = RingBuffer1D()
        self.fs = 8000
        
        self.max_threads = max_threads
        self.active_threads = []
        
        self.last_calculated_index = 0
        self.delay_seconds = 5  # default delay of 5 seconds
        
        self.events = []
        self.freq_threshold = 0.2

    def update_delay(self, delay):
        self.delay_seconds = delay

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
        #print("calculating")
        data = self.buffer.get_last_seconds(self.delay_seconds)
        #if len(data) < self.delay_seconds * self.fs:
        #    return  # Not enough new data
        frequencies, times, Sxx = self.compute_padded_spectrogram(signal)
        
        #print(np.shape(Sxx))
        self.buffer.add(Sxx)  # Add new spectrogram to the buffer
        self.detect_strong_signal(frequencies, times, Sxx)
        #self.callback(frequencies, times, self.buffer.get_all())  # Pass concatenated data
        
    def compute_padded_spectrogram(self, signal, fs=8000, window_length_s=5, overlap_s=2.5, target_duration_s=120):
        window_length = window_length_s * fs  # in samples
        overlap = overlap_s * fs  # in samples
        step_size = window_length - overlap
        total_duration_samples = target_duration_s * fs

        # Calculate number of windows needed
        number_of_windows = int((total_duration_samples - window_length) / step_size + 1)

        # Calculate exact length required
        required_length = (number_of_windows - 1) * step_size + window_length

        # Check if padding is necessary
        if len(signal) < required_length:
            padding_length = required_length - len(signal)
            signal = np.append(signal, np.zeros(padding_length))  # Pad with zeros

        # Calculate the spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs, window='hann', nperseg=window_length, noverlap=overlap, mode='magnitude')
        return (f, t, Sxx)
        
        
    def detect_strong_signal(self, frequencies, times, Sxx):
        freq_index = np.where((frequencies >= 78) & (frequencies <= 82))[0]
        if freq_index.size > 0:
            # Check for strong signal in the specified frequency range
            significant = np.max(Sxx[freq_index], axis=0) > self.freq_threshold  # Define your threshold
            for i, is_significant in enumerate(significant):
                if is_significant:
                    event_time = times[i] + (time.time() - self.buffer.size / self.fs)
                    self.events.append(event_time)  # Store event times
                    print('new event!')

class SignalApp:
    """Application class with synchronized time window updates for plots."""

    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Signal Analysis App")

        self.signal_generator = SignalGenerator(self.on_new_signal_data)
        threading.Thread(target=self.signal_generator.add_to_signal, daemon=True).start()

        self.spectrogram_calculator = SpectrogramCalculator()
        self.start_time = time.time()  # Record the start time of the application
        
        self.setup_plots()
        self.setup_controls()

    def setup_plots(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.ax1.set_title('Real-Time Signal with Noise')
        self.ax2.set_title('Stitched Real-Time Spectrogram')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=6, sticky='nsew')
        # Initialize a placeholder for the spectrogram plot
        self.pcm = None#self.ax2.pcolormesh(np.zeros((129,1000)), norm=Normalize(vmin=-40, vmax=0))  # Adjust size and limits as needed
        
        
        # Animation
        self.update_interval = 1000 #ms
        self.animation_save_ct = 5
        self.anim = animation.FuncAnimation(self.fig, self.update_plots, interval=self.update_interval, save_count=self.animation_save_ct, blit=False)

    def update_plots(self, frame):
        #print('updating plots')
        current_time = time.time()
        elapsed_time = current_time - self.start_time  # Time elapsed since start

        signal = np.array(self.signal_generator.buffer.get_all())
        times = np.linspace(elapsed_time - len(signal) / 800, elapsed_time, num=len(signal))

        self.ax1.clear()
        self.ax1.plot(times, signal)

        spectrogam_buffer = self.spectrogram_calculator.buffer
        #spectrogam_data = spectrogam_buffer.data
        spectrogam_data = spectrogam_buffer[-1]
        if self.pcm:
            pass    
        else:
            self.pcm = self.ax2.pcolormesh(spectrogam_data, norm=Normalize(vmin=-40, vmax=0))  # Adjust size and limits as needed
        '''
        try:
        
            self.pcm.set_array(np.ravel(spectrogam_data[:-1, :-1]))
            print('set array worked@')
        except:
            self.ax2.pcolormesh(10 * np.log10(spectrogam_data), shading='gouraud',cmap='viridis')
        '''
        '''
        print(spectrogam_data)
        spectrogam_shape = np.shape(spectrogam_data)
        #for i in range(1000-spectrogam_shape[1]):
        #    spectrogam_buffer.add(np.zeros(spectrogam_shape[0]))
        #spectrogam_data = spectrogam_buffer.get_all()
        padded_data = np.zeros((129, 1000 - spectrogam_shape[1]))
        padded_spectrogam = np.append(spectrogam_data[-1], padded_data)
        #padded_shape = np.shape(padded_spectrogam)
        current_spectrogram = self.pcm.get_array()
        #current_cropped = current_spectrogram[:,:spectrogam_shape]
        print(np.shape(padded_spectrogam), np.shape(current_spectrogram))
        if np.array_equal(current_spectrogram, padded_spectrogam):
            print('equal, skpping')
            pass
        else:
            try:
                self.pcm.set_array(np.ravel(spectrogam_data[:-1, :-1]))
                print('set array worked@')
            except:
                #pass
                print('improper Sxx size')
                #self.ax2.pcolormesh(np.zeros((spectrogam_shape[0]-1, 1000-1)), shading='gouraud',cmap='viridis')
                self.pcm = self.ax2.pcolormesh(spectrogam_data, norm=Normalize(vmin=-40, vmax=0))  # Adjust size and limits as needed
        '''
        
        return self.ax1, self.ax2

    def on_new_signal_data(self, signal):
        print('new signal to calculate!')
        # Trigger a new calculation each update
        #signal = self.signal_generator.last_signal_block
        self.spectrogram_calculator.calculate_spectrogram(signal)

    def update_spectrogram_plot(self, frequencies, times, Sxx):
        pass
        '''
        """Update the spectrogram plot with stitched data."""
        print("updating")
        print(np.shape(times), np.shape(frequencies), np.shape(Sxx))
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        real_times = times + elapsed_time - times[-1]  # Adjust times to sync with real-time

        # Determine if padding is needed
        last_time_point = real_times[-1]
        if elapsed_time > last_time_point:
            padding_length = int(self.spectrogram_calculator.fs * (elapsed_time - last_time_point))
            padding_array = np.zeros((frequencies.size, padding_length))
            Sxx = np.hstack((Sxx, padding_array))  # Append null data to the end of the spectrogram matrix
            extended_times = np.linspace(last_time_point, elapsed_time, num=padding_length)
            real_times = np.concatenate((real_times, extended_times))
            time_start = 0
            time_end = 20
            freq_start = 0
            freq_end = 4000
            real_times = np.linspace(time_start, time_end, np.shape(Sxx)[0]+1)
            frequencies = np.linspace(freq_start, freq_end, np.shape(Sxx)[1]+1)
        
        self.ax2.clear()
        max_size = 1028
        #print(shape(Sxx_stitched))
        #Sxx_stitched = Sxx_stitched[:-max_size]
        #try:
        #    self.pcm.set_array(np.ravel(Sxx[:-1, :-1]))
        #except:
        #    print(np.shape(times), np.shape(real_times), np.shape(frequencies), np.shape(Sxx))
        #    #self.ax2.pcolormesh(real_times, frequencies, 10 * np.log10(Sxx), shading='gouraud',cmap='viridis')
        #    self.ax2.pcolormesh(10 * np.log10(Sxx), shading='gouraud',cmap='viridis')
        '''

        self.canvas.draw_idle()

    def setup_controls(self):
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=0, sticky='ew')
        control_frame.grid_columnconfigure(0, weight=1)

        self.window_size_slider = tk.Scale(control_frame, from_=100, to_=1000, label="Window Size", orient=tk.HORIZONTAL)
        self.window_size_slider.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        self.window_size_slider.bind("<Motion>", self.adjust_window_size)
        
        self.delay_slider = tk.Scale(control_frame, from_=1, to_=10, label="Spectrogram Delay (s)", orient=tk.HORIZONTAL)
        self.delay_slider.set(5)  # Default value
        self.delay_slider.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        self.delay_slider.bind("<Motion>", self.adjust_delay)

    def adjust_delay(self, event=None):
        new_delay = self.delay_slider.get()
        self.spectrogram_calculator.update_delay(new_delay)

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
