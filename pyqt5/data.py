import numpy as np
import time
import scipy.signal as sig

'''
class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros(size)
        #self.index = 0

    def append(self, data):
        self.buffer = self.buffer
        self.buffer[:-len(data)] = data[:]
        #self.index += 1

    def get_all(self):
        return self.buffer
'''
class RingBuffer2D:
    def __init__(self, buffer_shape):
        """
        Initialize a 2D ring buffer.
        
        Parameters:
        - buffer_shape: tuple of integers, shape of the buffer (rows, columns).
          Rows represent the time dimension, and columns represent the frequency bins.
        """
        self.buffer_shape = buffer_shape
        self.buffer = np.zeros(buffer_shape, dtype=float)
        self.head = 0  # This will always point to the index of the oldest data

    def roll(self, shift):
        """
        Roll the buffer forward by 'shift' rows, dropping the oldest data.

        Parameters:
        - shift: integer, number of rows to roll (shift new data into the buffer).
        """
        if shift > 0:
            self.buffer[:-shift] = self.buffer[shift:]  # Shift data upwards
            self.buffer[-shift:] = 0  # Clear the newly vacated rows

    def add(self, new_data):
        """
        Add new spectrogram data to the buffer.

        Parameters:
        - new_data: 2D numpy array, the new spectrogram data to add. The number of columns must match the buffer's column count.
        """
        if new_data.shape[0] != self.buffer_shape[0]:
            raise ValueError("New data must have the same number of columns as the buffer.")
        
        new_rows = new_data.shape[1]
        self.roll(new_rows)
        self.buffer[-new_rows:] = new_data

    def get_all(self):
        """
        Retrieve all the data in the buffer.

        Returns:
        - numpy array: the full buffer content.
        """
        return self.buffer

    def __repr__(self):
        """
        String representation for debugging.
        """
        return f"RingBuffer2D(buffer_shape={self.buffer_shape}, buffer=\n{self.buffer})"

class RingBuffer:
    def __init__(self, size, dimension=1):
        """
        Initialize the Ring Buffer.
        
        Parameters:
        - size: integer or tuple, size of the buffer. If tuple, creates a 2D buffer.
        - dimension: integer, 1 for 1D buffer, 2 for 2D buffer with shape specified by size.
        """
        self.dimension = dimension
        if isinstance(size, tuple):
            self.buffer = np.zeros(size, dtype=float)  
        else:
            self.buffer = np.zeros(size, dtype=float) if dimension == 1 else np.zeros((size, 1), dtype=float)
        self.size = self.buffer.shape[0]
        self.width = self.buffer.shape[1] if dimension == 2 else None

    def roll(self, shift):
        """
        Roll the buffer to the front by 'shift' elements.

        Parameters:
        - shift: integer, number of elements to shift/roll.
        """
        self.buffer = np.roll(self.buffer, -shift, axis=0)

    def add(self, data):
        """
        Adds new data to the buffer, rolling forward by the length of the new data.
        
        Parameters:
        - data: ndarray, new data to add. Must match the buffer's second dimension if 2D.
        """
        data = np.asarray(data)
        if self.dimension == 2 and data.ndim == 1:
            data = data.reshape(-1, 1)
        
        #if data.shape[-1] != self.buffer.shape[-1]:
        #    raise ValueError("Data dimension does not match buffer width.")
        
        num_new_rows = data.shape[0]
        self.roll(num_new_rows)
        if self.dimension == 2:
            self.buffer[-num_new_rows:, :] = data
        else:
            self.buffer[-num_new_rows:] = data

    def get_all(self):
        """
        Retrieve all elements in the buffer.
        
        Returns:
        - buffer: ndarray, all data in the buffer.
        """
        return self.buffer

    def __repr__(self):
        """
        String representation of the current buffer state for debugging.
        """
        return f"RingBuffer(size={self.size}, dimension={self.dimension}, buffer=\n{self.buffer})"

class DataHandler:
    def __init__(self):
        self.fs = 8000
        self.buffer_len_s = 20
        self.feature_len_s = 5
        self.feature_overlap_s = 0.2
        self.nfft = 64
        self.noverlap = round(self.nfft/2)
        #self.buffers = [RingBuffer(1024) for _ in range(4)]
        self.rawDataBuffer = RingBuffer(self.fs*self.buffer_len_s)
        self.filteredDataBuffer = RingBuffer(self.fs*self.buffer_len_s)
        defaultSpectrogram = self.calc_spectrogram(self.rawDataBuffer.get_all())
        self.spectrogramBuffer = RingBuffer2D(buffer_shape=np.shape(defaultSpectrogram))
        self.num_features_per_buffer = round(self.buffer_len_s/self.feature_overlap_s)
        self.probabilityBuffer = RingBuffer(self.num_features_per_buffer)
        self.featureBuffer = RingBuffer(self.num_features_per_buffer)
        

    def poll_data(self):
        sleep_dur_ms = 200
        while True:
            
            # Simulate data polling
            new_data = np.random.rand(round(self.fs*sleep_dur_ms/1000))  # Simulated data for each plot

            self.rawDataBuffer.add(new_data)
            time.sleep(sleep_dur_ms/1000)

    def get_data(self):
        return [
            self.filteredDataBuffer.get_all(),
            self.spectrogramBuffer.get_all(),
            self.probabilityBuffer.get_all(),
            self.featureBuffer.get_all(),
            ]
    
    def process_data(self):
        data = self.rawDataBuffer.buffer[:200]
        # Apply high-pass filter
        filtered_data = self.high_pass_filter(data)
        self.filteredDataBuffer.add(filtered_data)

        # Calculate spectrogram
        Sxx = self.calc_spectrogram(filtered_data)
        self.spectrogramBuffer.add(Sxx)
    
    def high_pass_filter(self, data, cutoff=10.0, order=5):
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
        return sig.filtfilt(b, a, data)

    def calc_spectrogram(self, data):
        f, t, Sxx = sig.spectrogram(data, fs=self.fs, nfft=self.nfft, noverlap=self.noverlap, nperseg=self.nfft)
        return Sxx

