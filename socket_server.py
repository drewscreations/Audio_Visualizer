import socket
import pickle
import time
import threading

def start_server():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = 'localhost'
    port = 6000
    server_socket.bind((host, port))

    # Listen for client connection
    server_socket.listen(1)
    print("Server is listening on port", port)

    # Establish connection with client.
    conn, addr = server_socket.accept()
    print('Connected by', addr)

    try:
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
