import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram
import time

class DynamicSpectrogram:
    def __init__(self, fs, window_length=5, overlap_length=2.5):
        self.fs = fs
        self.window_length = int(window_length * fs)  # Convert seconds to samples
        self.overlap_length = int(overlap_length * fs)  # Convert seconds to samples
        self.step_size = self.window_length - self.overlap_length
        self.spectrogram_data = np.zeros((self.window_length // 2 + 1, 0))  # Initialize empty spectrogram data
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.spectrogram_data, aspect='auto', origin='lower', extent=[0, 1, 0, fs/2])
        self.last_time = time.time()

    def update(self, new_data):
        # This function would be called with new data
        # Calculate spectrogram of the new data
        frequencies, times, Sxx = spectrogram(new_data, fs=self.fs, nperseg=self.window_length, noverlap=self.overlap_length)
        current_time = time.time()
        time_delta = current_time - self.last_time
        self.last_time = current_time

        # Update the x-axis extents by time delta
        x_extent = max(times) + time_delta

        # If no new data, append zeros
        if Sxx.size == 0:
            Sxx = np.zeros((self.window_length // 2 + 1, len(times)))

        # Append new spectrogram data
        self.spectrogram_data = np.hstack((self.spectrogram_data, Sxx))

        # Update the image
        self.im.set_data(self.spectrogram_data)
        self.im.set_extent([0, x_extent, 0, self.fs / 2])
        self.ax.figure.canvas.draw()

    def generate_signal(self, freq = None, length=None):
        """Generate a simple sinusoidal signal based on the initialized frequency and length."""
        if freq == None:
            freq = self.frequency
        if length == None:
            length = self.fs * 5
        t = np.linspace(0, freq * 2 * np.pi, length)
        return np.sin(t)

    def add_noise(self, signal, amplitude=1, noise_frequency=1, duration=1):
        """
        Add noise to the signal.
        :param signal: The original signal.
        :param amplitude: Amplitude of the noise.
        :param noise_frequency: Frequency of the noise.
        :param duration: Duration over which noise is active as a percentage of total signal length.
        """
        noise = amplitude * np.random.normal(size=int(self.window_length * (duration)))
        full_noise = np.zeros_like(signal)
        full_noise[:len(noise)] = noise  # Apply noise at the start of the signal
        return signal + full_noise

    def run(self):
        # Simulation of data update
        def animate(i):
            signal = self.generate_signal(800, self.fs * 20)
            amp = np.random.randint(0, 10)/10
            freq = np.random.randint(0, 10)/10
            dur = np.random.randint(0, 10)/10
            #signal = self.add_noise(signal, amp, freq, dur)
            new_data = np.random.randn(self.fs * 5)  # Simulating 5 seconds of data
            self.update(signal)
        self.animation_save_ct = 5
        ani = FuncAnimation(self.fig, animate, interval=1000, save_count=self.animation_save_ct)
        plt.show()

# Usage
ds = DynamicSpectrogram(fs=8000)
ds.run()
