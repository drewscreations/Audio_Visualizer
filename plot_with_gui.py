import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
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
    def __init__(self):
        self.sample_rate = 8000 #kHz
        self.audio_buffer_duration_s = 20
        self.update_period_s = 2
        
        self.audio_buffer = RingBuffer(self.sample_rate*self.audio_buffer_duration_s)#np.zeros(self.sample_rate * self.audio_buffer_duration_s)
        self.time_buffer = RingBuffer(self.sample_rate*self.audio_buffer_duration_s)#np.zeros(self.sample_rate * self.audio_buffer_duration_s)
        self.start_time = time.time()
        
    def update_audio_buffer(self, noise = None):
        num_samples = int(self.update_period_s * self.sample_rate)
        t = np.linspace(0, self.update_period_s, num_samples, endpoint=False)
        time_offset = time.time()-self.start_time
        simulated_audio_freq = 0.1
        simulated_time = t + time_offset
        simulated_audio = 0.05 * np.sin(2 * np.pi * simulated_audio_freq * simulated_time)
        if noise:
            noise_duration = noise['duration']
            noise_amplitude = noise['amplitude']
            duration_samples = min(int(noise_duration*len(simulated_audio)), len(simulated_audio))  # Duration in samples
            start_index = np.random.randint(0, len(simulated_audio) - duration_samples)
            noise = np.random.normal(0, noise_amplitude*0.5, duration_samples)
            simulated_audio[start_index:start_index + duration_samples] += noise
        self.time_buffer.append(simulated_time)
        self.audio_buffer.append(simulated_audio)
    
    def audio_loop(self, noise):
        self.update_audio_buffer(noise)
        
    def get_data(self, num_samples):
        return [self.time_buffer.read(num_samples), self.audio_buffer.read(num_samples)]
        
class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Plot with Socket Data")

        self.max_win_size_s = 60
        self.max_scale = 10
        self.create_noise_controls()

        self.audio_simulator = Audio_Simulator()
        # Thread for reading socket data
        self.audio_thread = threading.Thread(target=self.audio_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Label to display socket data
        self.label = ttk.Label(root, text="Socket Data: Not Received Yet")
        self.label.pack()

        # Set up the plot
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot([], [], 'r-')        
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

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
        
        # Animation
        self.update_interval = 100 #ms
        self.animation_save_ct = 100
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=self.update_interval, save_count=self.animation_save_ct)

    def read_socket_data(self):
        while True:
            data = self.sock.recv(1024)
            if not data:
                pass
            else:# Unpickle the received data
                received_data = pickle.loads(data)
                print(received_data)
                self.label.config(text=f"Socket Data: {received_data}")
            time.sleep(1)

    def audio_loop(self):
        while True:
            if self.add_noise:
                noise = {}
                noise['duration'] = self.noise_duration.get()/(self.max_scale+1)
                if self.random_noise_amplitude.get():
                    noise['amplitude'] = np.random.randint(0, 10)/(self.max_scale+1)
                else:
                    noise['amplitude'] = self.noise_amplitude.get()/(self.max_scale+1)
            else:
                noise = None    
            self.audio_simulator.audio_loop(noise = noise)
            time.sleep(self.audio_simulator.update_period_s)

    def create_noise_controls(self):
        # Button to toggle noise addition
        self.add_noise = False
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

    def toggle_noise(self):
        self.add_noise = not self.add_noise
        self.noise_button.config(text="Remove Noise" if self.add_noise else "Add Noise")

    def update_plot(self, frame):
        # Determine the number of samples to display
        samples_to_display = int(self.window_size.get() * self.audio_simulator.sample_rate)
        samples_to_display = min(len(self.audio_simulator.audio_buffer.buffer), samples_to_display)
        [time_buffer, data_buffer] = self.audio_simulator.get_data(self.audio_simulator.sample_rate*20)
        display_data = data_buffer
        display_time = time_buffer
        t = np.linspace(0, self.window_size.get(), len(display_data))
        self.line.set_data(display_time, display_data)
        #print(f"dispaly time: {display_time[0]}, {display_time[-1000]} {display_time[-10]}, {len(display_time)}")
        current_time = time.time()
        time_offset = current_time-self.audio_simulator.start_time
        self.ax.set_xlim(time_offset - self.window_size.get(), time_offset+2)
        self.ax.set_ylim(-2, 2)
        return self.line,
    
    def toggle_animation(self):
        if self.anim_running:
            self.ani.event_source.stop()
            self.button.config(text='Start')
        else:
            self.ani.event_source.start()
            self.button.config(text='Stop')
        self.anim_running = not self.anim_running

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
