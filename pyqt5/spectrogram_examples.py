import numpy as np
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QApplication
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import sys
from scipy.signal import ShortTimeFFT, convolve
from scipy.signal.windows import gaussian



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.fs = 8000
        self.dur_s = 20
        
        # Define the signal
        #T_x, N = 1 / self.fs, round(self.fs*self.dur_s)  # 20 Hz sampling rate for 50 s signal
        #t_x = np.arange(N) * T_x  # time indexes for signal
        #f_i = 1000 * np.arctan((t_x - t_x[100]) / 2) + 500  # varying frequency
        #x = np.sin(2*np.pi*np.cumsum(f_i)*T_x) # the signal
        frequencies = [1000, 2000]  # in Hz
        x, t = self.convolve_waves(frequencies, self.fs, self.dur_s)

       

        # Plot configuration
        self.graphWidget.setLabels(left='Frequency (Hz)', bottom='Time (s)')
        self.graphWidget.setTitle("STFT with Gaussian Window")
        #self.graphWidget.plot(t, x*100, pen='r')

        # Create the image plot for the spectrogram
        x, t = self.get_new_signal(20)
        self.Sx = self.calc_spectrogram(x)
        self.img = pg.ImageItem(image=abs(np.transpose(self.Sx)))
        self.graphWidget.addItem(self.img)
        self.img.setLookupTable(pg.colormap.get('viridis').getLookupTable())
        self.img.setRect(pg.QtCore.QRectF(0, 0, self.dur_s, round(self.fs/2)))

        # Configuration to adjust image scaling and axes
        self.graphWidget.setXRange(0, self.dur_s)
        self.graphWidget.setYRange(0, 5000)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.update_speed_ms = 200
        self.timer.start(self.update_speed_ms)  # Update every 200 ms
        
    
    def convolve_waves(self, frequencies, sample_rate, duration):
        """
        Convolve multiple sine waves with specified frequencies.
        
        Parameters:
            frequencies (list of float): Frequencies of the waves to be convolved.
            sample_rate (int): Sample rate in samples per second.
            duration (float): Duration of each wave in seconds.
            
        Returns:
            numpy.ndarray: Result of convolving all the sine waves.
        """
        # Start with a delta function to keep the initial wave unchanged in the first convolution.
        result_wave = np.zeros(int(sample_rate * duration))
        result_wave[0] = 1  # delta function
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate and convolve each wave
        for freq in frequencies:
            sine_wave = np.sin(2 * np.pi * freq * t)
            result_wave = convolve(result_wave, sine_wave, mode='full')[:len(result_wave)]
        
        return result_wave, t
        
    def calc_spectrogram(self, data):
        g_std = 8  # standard deviation for Gaussian window in samples
        w_len = 100
        mfft = max(w_len, 200)
        w = gaussian(w_len, std=g_std, sym=True)  # symmetric Gaussian window
        SFT = ShortTimeFFT(w, hop=10, fs=self.fs, mfft=mfft, scale_to='magnitude')
        Sx = SFT.stft(data)  # perform the STFT
        return abs(Sx.T)
        
    def get_spectrogram_data(self, new_signal):
        new_data = self.calc_spectrogram(new_signal)
        data_shape = np.shape(new_data)
        self.Sx = np.roll(self.Sx, -data_shape[0], axis=0)
        self.Sx[-data_shape[0]:,:] = new_data
        return self.Sx
        
    def get_new_signal(self, dur):
        frequencies = [1000, 3000]# * round(5*np.random.random(1)[0])  # in Hz
        x, t = self.convolve_waves(frequencies, self.fs, dur)
        return x, t
        
    def update_plot(self):
        x, t = self.get_new_signal(self.update_speed_ms/1000)
        Sx = self.get_spectrogram_data(x)
        #img = pg.ImageItem(image=Sx)
        self.img.setImage(Sx)
        #self.graphWidget.addItem(img)
       
        self.img.setRect(pg.QtCore.QRectF(0, 0, self.dur_s, round(self.fs/2)))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())