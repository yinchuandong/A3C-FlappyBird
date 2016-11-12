import socket
import numpy as np
import cPickle


class GameState:
    def __init__(self, index=0, host='localhost', port=9600):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.reset()
        return

    def frame_step(self, input_actions):
        sock = self.sock
        sock.sendto(cPickle.dumps(input_actions), (self.host, self.port))

        header = sock.recv(1000)
        header = cPickle.loads(header)
        # print header

        data = str()
        buffer_size = header["buffer_size"]
        total_size = header["total_size"]
        block_num = header["block_num"]
        for i in range(block_num):
            receive_size = total_size - len(data)
            receive_size = receive_size if receive_size < buffer_size else buffer_size
            data += sock.recv(receive_size)
        data = cPickle.loads(data)
        return data

    def reset(self):
        action = np.zeros([2])
        action[0] = 1
        x_t, reward, terminal = self.frame_step(action)
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        self.reward = reward
        self.terminal = terminal
        return

    def process(self, actionId):
        action = np.zeros([2])
        action[actionId] = 1
        x_t1, reward, terminal = self.frame_step(action)
        x_t1 = np.reshape(x_t1, (84, 84, 1))
        self.s_t1 = np.append(self.s_t[:, :, 1:], x_t1, axis=2)
        self.reward = reward
        self.terminal = terminal
        return

    def update(self):
        self.s_t = self.s_t1
        return


if __name__ == '__main__':
    gamestate = GameState()
    for i in range(200):
        gamestate.process(0)
