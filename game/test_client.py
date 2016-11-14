import socket
import sys
import numpy as np
import cPickle
import time


HOST, PORT = "localhost", 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def main():
    # data = " ".join(sys.argv[1:])
    # send action
    sock.sendto(str(1), (HOST, PORT))
    
    header = sock.recv(1000)
    header = cPickle.loads(header)
    print header

    data = str()
    buffer_size = header["buffer_size"]
    total_size = header["total_size"]
    block_num = header["block_num"]
    for i in range(block_num):
        receive_size = total_size - len(data)
        receive_size = receive_size if receive_size < buffer_size else buffer_size
        data += sock.recv(receive_size)
    data = cPickle.loads(data)

    # print "Sent:     {}".format(data)
    print "Received: {}".format(np.shape(data))

    return


if __name__ == '__main__':
    for i in range(100):
        main()
        # time.sleep(1 / 30.0)
