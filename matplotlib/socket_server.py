import socket
import pickle
import time
import threading
import random

shouldRestart = True


class SignalGenerator:
    """Handles the generation and manipulation of signals."""

    def __init__(self, length=1000, frequency=5):
        """
        Initialize default parameters for the signal.
        :param length: Number of data points in the signal.
        :param frequency: Frequency of the base signal.
        """
        self.length = length
        self.frequency = frequency

    def generate_signal(self):
        """Generate a simple sinusoidal signal based on the initialized frequency and length."""
        t = np.linspace(0, self.frequency * 2 * np.pi, self.length)
        return np.sin(t)

    def add_noise(self, signal, amplitude=1, noise_frequency=1, duration=100):
        """
        Add noise to the signal.
        :param signal: The original signal.
        :param amplitude: Amplitude of the noise.
        :param noise_frequency: Frequency of the noise.
        :param duration: Duration over which noise is active as a percentage of total signal length.
        """
        noise = amplitude * np.random.normal(size=int(self.length * (duration / 100.0)))
        full_noise = np.zeros_like(signal)
        full_noise[:len(noise)] = noise  # Apply noise at the start of the signal
        return signal + full_noise

def start_server():
    # Create a socket object
    
    host = 'localhost'
    port = 6000
    

 
    while shouldRestart:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            # Listen for client connection
            server_socket.listen(1)
            print("Server is listening on port", port)

            # Establish connection with client.
            conn, addr = server_socket.accept()
            print('Connected by', addr)
            while True:
                # Data to send
                data_to_send = {'time': time.time(), 'message': 'Hello, Client!'}
                # Pickle the dictionary
                pickled_data = pickle.dumps(data_to_send)
                
                # Send pickled data
                
                conn.sendall(pickled_data)
                print("Sent:", data_to_send)
                
                # Wait for 0.5 seconds
                time.sleep(0.5)
        except Exception as e:
            print("Server error:", e)
            time.sleep(1)
        finally:
            # Close the connection
            conn.close()
            server_socket.close()
            print("Server closed.")

def start_client():
    time.sleep(1)  # Delay to allow server to start first
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = 'localhost'
    port = 12345

    # Connect to the server
    client_socket.connect((host, port))
    print("Client connected to", host, "on port", port)

    try:
        while True:
            # Receive data
            data = client_socket.recv(1024)
            if not data:
                break
            # Unpickle the received data
            received_data = pickle.loads(data)
            print("Received:", received_data)
    except Exception as e:
        print("Client error:", e)
    finally:
        # Close the connection
        client_socket.close()
        print("Client closed.")

if __name__ == "__main__":
    # Start server and client threads
    start_server()
    #server_thread = threading.Thread(target=start_server)
    #client_thread = threading.Thread(target=start_client)

    #server_thread.start()
    #client_thread.start()

    #server_thread.join()
    #client_thread.join()
