from multiprocessing import Process, Pipe


def f(conn):
    cmd = conn.recv()
    if cmd == 'process':
        conn.send(['img', 'reward', 'terminal'])
    else:
        conn.send([-1, -1, -1])
    # conn.close()


if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    parent_conn.send('process')
    # parent_conn.send('hello')
    print 'after child close'
    print parent_conn.recv()   # prints 
    # print parent_conn.recv()   # prints 
    p.join()