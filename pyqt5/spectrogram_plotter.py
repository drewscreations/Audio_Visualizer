import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QComboBox
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from scipy.signal import ShortTimeFFT, convolve
from scipy.signal.windows import gaussian
import pyaudio

class RollingBuffer:
    def __init__(self, shape, fs=8000):
        self.buffer_shape = shape
        self.fs = fs
        self.buffer = np.zeros(shape)
        
    def randomize_buffer(self):
        self.buffer = np.zeros(self.buffer_shape)
    
    def add(self, new_data):
        data_shape = np.shape(new_data)
        num_cols = data_shape[0]
        self.buffer = np.roll(self.buffer, -num_cols, axis=0)
        try:
            num_rows = data_shape[1]
            self.buffer[-num_cols:, :] = new_data
        except IndexError:
            self.buffer[-num_cols:] = new_data
        
    def get_last_s(self, s):
        num_samples = round(s*self.fs)
        try:
            cropped_data = self.buffer[-num_samples:,:]
        except IndexError:
            cropped_data = self.buffer[-num_samples:]
        return cropped_data

class AudioStreamHandler(QThread):
    data_ready = pyqtSignal(np.ndarray)

    def __init__(self, stream_index, sample_rate=8000, chunk_size=1024, parent=None):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream_index = stream_index
        self.running = False

        self.p = pyaudio.PyAudio()
        device_info = self.p.get_device_info_by_index(self.stream_index)
        print(device_info)
        self.channels = min(device_info.get('maxInputChannels', 1), 1)
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  input=True,
                                  input_device_index=self.stream_index,
                                  frames_per_buffer=self.chunk_size)

    def run(self):
        self.running = True
        while self.running:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.data_ready.emit(audio_data)

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class SignalGenerator:
    def __init__(self, frequency, sample_rate, amplitude):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.amplitude = amplitude

    def generate(self, dur_s, add_noise=False, noise_freq=0):
        length = round(dur_s * self.sample_rate)
        t = np.arange(length) / self.sample_rate
        signal = np.sin(2 * np.pi * self.frequency * t)
        if add_noise:
            noise = 0.5 * np.sin(2 * np.pi * noise_freq * t + np.random.random(size=length))
            signal += noise
        return signal * self.amplitude

class SpectrogramViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.generator = SignalGenerator(frequency=1000, sample_rate=8000, amplitude=1)
        self.audio_handler = None
        self.fs = self.generator.sample_rate
        self.plot_update_ms = 100
        self.data_update_ms = 50
        self.plot_window_len_s = 20
        self.total_samples = round(self.fs * self.plot_window_len_s)
        self.noverlap = 10
        self.use_noise = False
        self.noise_freq = 100
        self.sfft_buffer = RollingBuffer((200, 500))  # Assuming 500 is the FFT size
        self.raw_data_buffer = RollingBuffer(self.total_samples)
        self.initUI()
        self.list_audio_devices()
        

    def initUI(self):
        self.setWindowTitle("Spectrogram From Stream")
        self.setGeometry(2540, 100, 600, 400)
        self.central_widget = QWidget()  # Central widget to hold the layout
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()  # Vertical layout
        self.central_widget.setLayout(self.layout)

        # Signal plot
        self.signal_plot_widget = pg.PlotWidget(name='SignalPlot')
        #elf.signal_plot_widget.setXRange(0, self.plot_window_len_s)
        self.signal_plot_widget.setLabel('bottom', 'Time', units='s')
        self.signal_curve = self.signal_plot_widget.plot(pen='y')
        self.layout.addWidget(self.signal_plot_widget)

        # Spectrogram plot
        self.spectrogram_plot_widget = pg.PlotWidget(name='SpectrogramPlot')
        self.spectrogram_plot_widget.setLabel('bottom', 'Time', units='s')
        self.spectrogram_image = pg.ImageItem()
        self.spectrogram_image.setLookupTable(pg.colormap.get('viridis').getLookupTable())
        self.spectrogram_plot_widget.addItem(self.spectrogram_image)
        self.layout.addWidget(self.spectrogram_plot_widget)

        # Ensure x-axes are linked
        self.signal_plot_widget.setXLink(self.spectrogram_plot_widget)
        data = self.generator.generate(self.plot_window_len_s, add_noise=True, noise_freq=100)
        self.Sx = self.calc_spectrogram(data) #(80009, 65)
        
        # Timer to update plots
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.plot_update_ms) 
        self.timer_running = True
        
        #self.dataStreamTimer = QTimer()
        #self.dataStreamTimer.timeout.connect(self.addNewData)
        #self.dataStreamTimer.start(self.data_update_ms)
        
        self.noise_toggle_timer = QTimer()
        self.noise_toggle_timer.timeout.connect(self.toggle_noise)
        self.noise_toggle_timer.start(3000)
        
        # Start/Stop Button
        self.toggle_button = QPushButton("Stop", self)
        self.toggle_button.clicked.connect(self.toggle_timer)
        self.layout.addWidget(self.toggle_button)
        
        self.device_combo = QComboBox()
        self.layout.addWidget(self.device_combo)

        self.start_button = QPushButton('Start Audio Stream')
        self.start_button.clicked.connect(self.start_audio_stream)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop Audio Stream')
        self.stop_button.clicked.connect(self.stop_audio_stream)
        self.layout.addWidget(self.stop_button)

    def toggle_noise(self):
        #self.use_noise = not self.use_noise
        rand_val = np.random.random()
        self.generator.frequency = round(4000 * rand_val)
        self.generator.amplitude = round(5 * rand_val)
        self.noise_freq = 1000 * np.random.random()

    def toggle_timer(self):
        if self.timer_running:
            self.timer.stop()
            self.toggle_button.setText("Start")
        else:
            self.timer.start(self.plot_update_ms)
            self.toggle_button.setText("Stop")
        self.timer_running = not self.timer_running

    def list_audio_devices(self):
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            self.device_combo.addItem(device_info['name'], i)
        p.terminate()

    def start_audio_stream(self):
        self.stop_audio_stream()
        stream_index = self.device_combo.currentData()
        self.audio_handler = AudioStreamHandler(stream_index)
        self.audio_handler.data_ready.connect(self.handle_audio_data)
        self.audio_handler.start()

    def stop_audio_stream(self):
        if self.audio_handler:
            self.audio_handler.stop()
            self.audio_handler = None

    def handle_audio_data(self, data):
        # Handle audio data (e.g., update plots)
        #print(audio_data)
        self.raw_data_buffer.add(data)
        self.get_spectrogram_data(data)

    def calc_spectrogram(self, data):
        g_std = 8  # standard deviation for Gaussian window in samples
        w_len = 20
        mfft = max(w_len, 128)
        w = gaussian(w_len, std=g_std, sym=True)  # symmetric Gaussian window
        SFT = ShortTimeFFT(w, hop=2, fs=self.fs, mfft=mfft, scale_to='magnitude')
        Sx = SFT.stft(data)  # perform the STFT
        return abs(Sx.T)
        
    def get_spectrogram_data(self, new_signal):
        new_data = self.calc_spectrogram(new_signal)
        data_shape = np.shape(new_data)
        self.Sx = np.roll(self.Sx, -data_shape[0], axis=0)
        self.Sx[-data_shape[0]:,:] = new_data
        return self.Sx

    def addNewData(self):
        data_dur = self.data_update_ms/1000
        data = self.generator.generate(data_dur, add_noise=self.use_noise, noise_freq=self.noise_freq)
        self.raw_data_buffer.add(data)
        self.get_spectrogram_data(data)

    def update(self):
        overlapped_data = self.raw_data_buffer.get_last_s(self.plot_update_ms/1000)
        overlap_num = round(self.fs*0.1)
        #self.signal_curve.setData(self.raw_data_buffer.get_last_s(self.plot_window_len_s))
        x = np.linspace(-self.plot_window_len_s, 0, self.total_samples)
        y = self.raw_data_buffer.buffer
        self.signal_curve.setData(x, y)
        Sx = self.Sx
        
        self.spectrogram_image.setImage(Sx, autoLevels=True)
        self.spectrogram_image.setRect(pg.QtCore.QRectF(-self.plot_window_len_s, 0, self.plot_window_len_s, round(self.fs/2)))

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SpectrogramViewer()
    viewer.show()
    sys.exit(app.exec_())
