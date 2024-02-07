import socket
import pandas as pd
import pickle
import os
import threading

def handle_client(c):
    data = b''
    while True:
        packet = c.recv(1024)
        if not packet: 
            break
        data += packet
    df, filename = pickle.loads(data)
    directory = '/path/to/your/directory' # change to correct directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(os.path.join(directory, f'data_exp_{filename}.csv'), index=False)
    c.close()

def main():
    s = socket.socket()
    s.bind(('127.0.0.1', 16666)) # change to correct server IPv4 address
    s.listen(5)

    while True:
        c, addr = s.accept()
        greeting_message = 'Connected to server...'
        c.send(greeting_message.encode('ascii'))
        threading.Thread(target=handle_client, args=(c,)).start()

if __name__ == "__main__":
    main()
