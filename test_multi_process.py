import signal
import os
from multiprocessing import Process, Value, Array
from threading import Thread
from game.game_state import GameState

stop_requested = False


def train_function(index):
    print 'train==', index
    game = GameState()
    for i in range(1000):
        if stop_requested:
            break
        game.process(0)
    return


def signal_handler(signal_, frame_):
    print 'You pressed Ctrl+C !'
    global stop_requested
    stop_requested = True
    return


def run():
    train_treads = []
    for i in range(2):
        train_treads.append(Process(target=train_function, args=(i,)))
        # train_treads.append(Thread(target=train_function, args=(i,)))

    signal.signal(signal.SIGINT, signal_handler)

    for t in train_treads:
        t.start()

    print 'Press Ctrl+C to stop'
    signal.pause()

    print 'Now saving data....'
    for t in train_treads:
        t.join()

    return


def flappy():
    game = GameState()
    # for i in range(100):
    # game.process(0)
    return


def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]


def test():
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    # flappy()
    # p = Process(target=f, args=(num, arr))
    p = Process(target=flappy)
    p.start()
    p.join()

    print num.value
    print arr[:]


if __name__ == '__main__':
    # test()
    run()
