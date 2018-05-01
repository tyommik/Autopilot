import socket
import sock_config

sock = socket.create_connection((sock_config.SERVER_ADDRESS, sock_config.SERVER_PORT))
sock.sendall("PING".encode('utf-8'))
sock.close()