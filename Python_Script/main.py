import socket
from time import sleep

host = "192.168.1.67"
port = 23
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))

x = 0
while x != "END": # TODO głupia małpo zamien na for
    x = input()
    s.sendall(bytes(str(x), 'utf-8'))
    sleep(1)
s.close()