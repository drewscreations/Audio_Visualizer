import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from scipy.signal import spectrogram
import numpy as np
import socket
import threading
import time
import pickle

class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=float)
        self.start = 0  # Points to the start of valid data
        self.end = 0  # Points to one past the last valid data item

    def append(self, data):
        data_length = len(data)
        # Check where to start writing new data
        if data_length > self.capacity:
            data = data[-self.capacity:]  # Only keep the last 'capacity' elements
            data_length = self.capacity

        end_pos = (self.end + data_length) % self.capacity
        
        if end_pos < self.end:
            # Data wraps around the buffer
            self.buffer[self.end:] = data[:self.capacity - self.end]
            self.buffer[:end_pos] = data[self.capacity - self.end:]
        else:
            # Data fits without wrapping
            self.buffer[self.end:end_pos] = data
        
        self.end = end_pos
        if data_length >= self.capacity or self.end < self.start:
            self.start = (self.end + 1) % self.capacity

    def read(self, size):
        # Assuming read is always valid and within buffer data limits
        if size > self.capacity:
            size = self.capacity

        start_index = (self.end - size) % self.capacity
        if start_index < self.end:
            return self.buffer[start_index:self.end]
        else:
            return np.concatenate((self.buffer[start_index:], self.buffer[:self.end]))

    def __str__(self):
        if self.start < self.end:
            return str(self.buffer[self.start:self.end])
        else:
            return str(np.concatenate((self.buffer[self.start:], self.buffer[:self.end])))


class Audio_Simulator:
    def __init__(self, add_noise, noise_duration, noise_amplitude, random_amplitude):
        self.sample_rate = 8000 #kHz
        self.audio_buffer_duration_s = 20
        self.update_period_s = 2
        self._on_new_data = None
        self.audio_buffer = RingBuffer(self.sample_rate*self.audio_buffer_duration_s)#np.zeros(self.sample_rate * self.audio_buffer_duration_s)
        self.time_buffer = RingBuffer(self.sample_rate*self.audio_buffer_duration_s)#np.zeros(self.sample_rate * self.audio_buffer_duration_s)
        self.noise = {}
        self.start_time = time.time()
        self.new_data = False
        self.add_noise_var = add_noise
        self.noise_duration_var = noise_duration
        self.noise_amplitude_var = noise_amplitude
        self.random_amplitude_var = random_amplitude
        
       
    def update_noise_parameters(self):
        add_noise = self.add_noise_var.get()
        noise_duration = 0
        noise_amplitude = 0
        if add_noise:
            noise_duration = self.noise_duration_var.get()/10
            random_amplitude = self.random_amplitude_var.get()
            if random_amplitude:
                noise_amplitude = np.random.randint(0, 10)/11
            else:
                noise_amplitude = self.noise_amplitude_var.get()/10
        return [add_noise, noise_duration, noise_amplitude]

        
    def update_audio_buffer(self, noise = None):
        [add_noise, noise_duration, noise_amplitude] = self.update_noise_parameters()
        num_samples = int(self.update_period_s * self.sample_rate)
        t = np.linspace(0, self.update_period_s, num_samples, endpoint=False)
        time_offset = time.time()-self.start_time
        simulated_audio_freq = 0.1
        simulated_time = t + time_offset
        simulated_audio = 0.05 * np.sin(2 * np.pi * simulated_audio_freq * simulated_time)
        if add_noise:
            duration_samples = min(int(noise_duration*len(simulated_audio)), len(simulated_audio)-1)  # Duration in samples
            start_index = np.random.randint(0, len(simulated_audio) - duration_samples)
            noise = np.random.normal(0, noise_amplitude*0.5, duration_samples)
            simulated_audio[start_index:start_index + duration_samples] += noise
        self.time_buffer.append(simulated_time)
        self.audio_buffer.append(simulated_audio)
        self.new_data = True

    
    def audio_loop(self):
        self.update_audio_buffer()
        time.sleep(self.update_period_s)
        
    def get_data(self, num_samples):
        return [self.time_buffer.read(num_samples), self.audio_buffer.read(num_samples)]
       
class AudioAnalyzer:
    def __init__(self, plotManager, audioSimulator):
        self.plotManager = plotManager
        self.audio_simulator = audioSimulator
        #self.audio_data = np.zeros(100)
        self.sample_rate = self.audio_simulator.sample_rate
        self.spectrogram_data = ([], [], [])
        self.spectrogam_window_size = 256  # Initial window size for the FFT
        self.spectrogam_window_s = 2
        
        #self._on_new_data = self.on_new_data
        #self._on_new_calculation = on_new_calculation

    def process_audio(self):
        while True:
            self.audio_simulator.update_audio_buffer()
            [time_buffer, data_buffer] = self.audio_simulator.get_data(self.sample_rate*self.plotManager.line_plot_window_size) #get 20s data
            self.audio_data = data_buffer
            self.plotManager.update_time_series_data(time_buffer, data_buffer)
            self.calculate_spectrogram()
            f, t, Sxx = self.spectrogram_data
            self.plotManager.update_spectrogram_data(f, t, Sxx)
            self.audio_simulator.new_data = False
            time.sleep(self.audio_simulator.update_period_s)
                

    #def update_audio_data(self, new_data):
    #    self.audio_data = new_data
    #    #self._on_new_data(self.audio_data)
    #    self.plotManager.update_time_series_data(new_data)

    def calculate_spectrogram(self, nperseg=None):
        if nperseg == None:
            nperseg = self.spectrogam_window_size
        f, t, Sxx = spectrogram(self.audio_data[:self.spectrogam_window_s*self.sample_rate], self.sample_rate, window='hamming', nperseg=nperseg)
        self.spectrogram_data = (f, t, Sxx)
        self.plotManager.update_spectrogram_data(f, t, Sxx)
            
    def get_spectrogam_data(self):
        return self.spectrogram_data

    
       
class PlotManager():
    def __init__(self):
        self.name = "PlotManager"
        
        self.start_time = time.time()
        self.fig, self.ax = plt.subplots(2, 1)
        self.time_series_plot = self.ax[0]
        self.spectrogram_plot = self.ax[1]
        self.line_plot_window_size = 20
        self.time_series_data = [[], []]
        
        self.line, = self.time_series_plot.plot([], [], 'r-')  
        self.spectrogram_window_sizes = [32, 256, 1024]
        self.spectrogram_window_size = self.spectrogram_window_sizes[1]
        self._initialize_spectrogam_plot()
        axcolor = 'lightgoldenrodyellow'
        ax_win = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
        #with np.errstate(divide='ignore'):
        #        self.spectrogam_image = self.spectrogram_plot.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        # Animation
        self.update_interval = 100 #ms
        self.animation_save_ct = 100
        
        self.line_animation = animation.FuncAnimation(self.fig, self._update_plots, interval=self.update_interval, save_count=self.animation_save_ct)
        #self.spectrogram_animation = animation.FuncAnimation(self.fig, self._update_spectrogam_plot, interval=self.update_interval, save_count=self.animation_save_ct)
        
        # Slider for adjusting window size
        
        
        

        self.slider = Slider(ax_win, 'Window Size',
                            self.spectrogram_window_sizes[0],
                            self.spectrogram_window_sizes[2],
                            valinit=self.spectrogram_window_size,
                            valstep=32)
        self.slider.on_changed(self.update_spectrogram_windowing_size)
        
    def update_time_series_data(self, x, y):
        self.time_series_data = [x,y]
        #self._update_time_series_plot

    def update_spectrogram_data(self, f, t, Sxx):
        self.spectrogram_data = (f, t, Sxx)

    def _update_plots(self, frame):
        self._update_line_plot(frame)
        self._update_spectrogam_plot(frame)

    def _update_line_plot(self, frame):
        [x, y] = self.time_series_data
        self.line.set_data(x, y)
        #print(f"dispaly time: {display_time[0]}, {display_time[-1000]} {display_time[-10]}, {len(display_time)}")
        current_time = time.time()
        time_offset = current_time-self.start_time
        self.time_series_plot.set_xlim(time_offset - self.line_plot_window_size, time_offset+2)
        self.time_series_plot.set_ylim(-2, 2)

        return self.line,
    
    def _initialize_spectrogam_plot(self):
            f, t, Sxx = spectrogram(np.ones(8000)*20, 8000, window='hamming', nperseg=self.spectrogram_window_size)
            self.spectrogram_data = (f, t, Sxx)
            self.spectrogram_plot.set_ylabel('Frequency [Hz]')
            self.spectrogram_plot.set_xlabel('Time [s]')
            self.spectrogram_plot.set_title('Spectrogram')
            #f, t, Sxx = self.spectrogram_data
            with np.errstate(divide='ignore'):
                self.spectrogam_image = self.spectrogram_plot.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    
    def _update_spectrogam_plot(self, frame):

            f, t, Sxx = self.spectrogram_data
            if np.isnan(Sxx).any():
                pass
            else:
                self.spectrogam_image.set_array(10 * np.log10(Sxx.ravel()))
                self.spectrogram_plot.relim()
            #self.spectrogram_plot.cla()
            #with np.errstate(divide='ignore'):
            #    audioAnalyzer
            #    self.spectrogram_plot.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            #self.spectrogram_plot.set_ylabel('Frequency [Hz]')
            #self.spectrogram_plot.set_xlabel('Time [s]')
            #self.spectrogram_plot.set_title('Spectrogram')

            #plt.draw()

    def update_spectrogram_windowing_size(self, val):
        pass
        #self.spectrogram_window_size = int(val)
        #self.update_spectrogam_plot() 
        
class AudioAnalyzingApp:
    def __init__(self, root):
        np.seterr(divide='ignore')
        self.root = root
        self.root.title("Real-time Plot with Socket Data")
        
        self.max_win_size_s = 60
        self.max_scale = 10
        #[add_noise, noise_duration, noise_amplitude] = self.create_noise_controls()
        noise_vars = self.create_noise_controls()

        self.audioSimulator = Audio_Simulator(*noise_vars)
        self.plotManager = PlotManager()
        
        # Thread for reading socket data

        
        # Label to display socket data
        self.label = ttk.Label(root, text="Socket Data: Not Received Yet")
        self.label.pack()

        # Set up the plot
        #self.fig, self.ax = plt.subplots(2, 1)
        #self.fig = Figure()
        #self.ax = self.fig.add_subplot(111)
        #self.time_series_plot = self.ax[0]
              
        self.audioAnalyzer = AudioAnalyzer(self.plotManager, self.audioSimulator)
        self.canvas = FigureCanvasTkAgg(self.plotManager.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        
        self.audio_thread = threading.Thread(target=self.audioAnalyzer.process_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        # Button to start/stop the animation
        self.anim_running = True
        self.button = ttk.Button(root, text="Stop", command=self.toggle_animation)
        self.button.pack()

        # Socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(('localhost', 6000))
        
        # Thread for reading socket data
        self.socket_thread = threading.Thread(target=self.read_socket_data)
        self.socket_thread.daemon = True
        self.socket_thread.start()

    def read_socket_data(self):
        while True:
            data = self.sock.recv(1024)
            if not data:
                pass
            else:# Unpickle the received data
                received_data = pickle.loads(data)
                #print(received_data)
                self.label.config(text=f"Socket Data: {received_data}")
            time.sleep(1)

   

    def create_noise_controls(self):
        # Button to toggle noise addition
        self.add_noise = tk.BooleanVar(value=False)
        self.noise_button = ttk.Button(self.root, text="Add Noise", command=self.toggle_noise)
        self.noise_button.pack()

        # Slider for noise duration
        self.noise_duration = tk.DoubleVar(value=self.max_scale/2)  # Default to 0.1 seconds
        self.noise_duration_slider = tk.Scale(self.root, from_=0, to=self.max_scale, variable=self.noise_duration, orient='horizontal', label="noise duration")
        self.noise_duration_slider.pack()

        # Slider for noise amplitude
        self.noise_amplitude = tk.DoubleVar(value=self.max_scale/2)  # Default to 0.1 amplitude
        self.noise_amplitude_slider = tk.Scale(self.root, from_=0, to=self.max_scale, variable=self.noise_amplitude, orient='horizontal', label="noise amplitude")
        self.noise_amplitude_slider.pack()
        
        self.random_noise_amplitude = tk.BooleanVar(value=False)
        self.random_noise_amplitude_cb = ttk.Checkbutton(self.root, variable=self.random_noise_amplitude, text="random noise amplitude")
        self.random_noise_amplitude_cb.pack()
        
        
        
        self.window_size = tk.DoubleVar(value=self.max_win_size_s/2)  # Default window size of 20 seconds
        self.window_size_slider = tk.Scale(self.root, from_=1, to=self.max_win_size_s, variable=self.window_size, orient='horizontal', label="window size")#, command=self.update_plot)
        self.window_size_slider.pack()

        return [self.add_noise, self.noise_duration, self.noise_amplitude, self.random_noise_amplitude]
    

    def toggle_noise(self):
        #noise = self.audioSimulator.add_noise
        #duration = self.noise_duration.get()
        #amplitude = self.noise_amplitude.get()
        #isRandomAmplitude = self.random_noise_amplitude
        #self.audioSimulator.set_noise_parameters(not noise, amplitude=amplitude, duration=duration, isRandomAmplitude=isRandomAmplitude)
        self.add_noise.set(not self.add_noise.get())
        self.noise_button.config(text="Remove Noise" if self.add_noise else "Add Noise")


    def toggle_animation(self):
        if self.anim_running:
            self.ani.event_source.stop()
            self.button.config(text='Start')
        else:
            self.ani.event_source.start()
            self.button.config(text='Stop')
        self.anim_running = not self.anim_running
        
    def main_loop(self):
        while True:
            # Determine the number of samples to display
            samples_to_display = int(self.window_size.get() * self.audio_simulator.sample_rate)
            samples_to_display = min(len(self.audio_simulator.audio_buffer.buffer), samples_to_display)
            [time_buffer, data_buffer] = self.audio_simulator.get_data(self.audio_simulator.sample_rate*20)
            display_data = data_buffer
            display_time = time_buffer
            t = np.linspace(0, self.window_size.get(), len(display_data))
            self.plotManager.update_time_series_data(time_buffer, data_buffer)
            #(f, t, Sxx) = self.audio_analyzer.get_spectrogam_data() 
            #self.plotManager.update_spectrogram_data(f, t, Sxx)
            time.sleep(0.1)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzingApp(root)
    app.root.mainloop()
