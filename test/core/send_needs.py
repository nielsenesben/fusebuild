"""Helper for test_main.py, stubbing runtarget.py"""

import socket
import sys
import time

socket_path = sys.argv[1]
client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client_socket.connect(socket_path)
main_connection_file = client_socket.makefile("rw")
main_connection_file.write("needs: //blocked //blocker hard\n")
main_connection_file.flush()
main_connection_file.readline()
