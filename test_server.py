from SocketServer import BaseRequestHandler, UDPServer
import numpy as np
from PIL import Image
import cPickle
import math
from game.game_state import GameState


gamestate = GameState()


class MyUDPHandler(BaseRequestHandler):

    def setup(self):
        # print '---------setup'
        # self.count = 1
        return

    def handle(self):
        socket = self.request[1]
        # image = Image.open('images/preprocess.png')
        # image = image.resize((120, 120), Image.ANTIALIAS)
        # image.save('images/test.png')
        # data = np.array(image)
        # data = cPickle.dumps(data)
        global gamestate
        gamestate.process(1)
        data = gamestate.s_t
        data = cPickle.dumps(data)

        # not larger than 8192 due to the limitation of MXU of udp
        buffer_size = 8192
        total_size = len(data)
        block_num = int(math.ceil(total_size / float(buffer_size)))

        # send the length
        offset = 0
        header = {
            "buffer_size": buffer_size,
            "total_size": total_size,
            "block_num": block_num
        }
        header = cPickle.dumps(header)
        socket.sendto(header, self.client_address)
        while offset < total_size:
            end = offset + buffer_size
            end = end if end < total_size else total_size
            socket.sendto(data[offset: end], self.client_address)
            offset = end

        # self.count += 1
        # print self.count
        return

    def finish(self):
        # print '---------finish'
        # self.count = 0
        # print self.count
        return


class MyUDPServer(UDPServer):
    def __init__(self, server_address, handler_class=MyUDPHandler):
        UDPServer.__init__(self, server_address, handler_class)
        return


if __name__ == "__main__":
    HOST, PORT = "localhost", 9998
    server = MyUDPServer((HOST, PORT), MyUDPHandler)
    server.serve_forever()

    # test()
