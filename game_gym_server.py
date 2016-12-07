import sys
import cPickle
import math

from SocketServer import BaseRequestHandler, UDPServer
from ple.games.flappybird import FlappyBird
from ple import PLE


flapp_bird = FlappyBird()
ple = PLE(flapp_bird, fps=30, display_screen=True)
print ple.getActionSet()
ple.init()


def test():
    global ple
    for i in range(300):
       if ple.game_over():
               ple.reset_game()
       observation = ple.getScreenRGB()
       # action = agent.pickAction(reward, observation)
       reward = ple.act(119)
    return


class UDPHandler(BaseRequestHandler):

    def handle(self):
        action = self.request[0]
        action = cPickle.loads(action)
        socket = self.request[1]

        global flapp_bird
        x_t, reward, terminal = flapp_bird.frame_step(action)
        data = cPickle.dumps((x_t, reward, terminal))

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

        return


class GameServer(UDPServer):
    def __init__(self, server_address, handler_class=UDPHandler):
        UDPServer.__init__(self, server_address, handler_class)
        return


# how to use:
# args: index, please be consistent for your a3c agent thread index
# python game_server.py 0
if __name__ == "__main__":
    # host, port = "localhost", 9600
    # if len(sys.argv) > 1:
    #     index = int(sys.argv[1])
    #     port = port + index
    # print port
    # server = GameServer((host, port), UDPHandler)
    # server.serve_forever()

    test()
