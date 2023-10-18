import pyhelib
import zmq
import zlib
import uuid
import time
import pickle
import signal
import threading

import numpy as np
from redis import StrictRedis, ConnectionPool

EXPONENT       = 11
POWER          = 1 << EXPONENT
PRODUCT_POWER  =  1 << (EXPONENT << 1)

# 192.168.122.30
#PEER_ZMQ_URL   = 'tcp://192.168.122.31:3333'
#SERVER_ZMQ_URL = 'tcp://192.168.122.30:3333'
#PARTY_ID       = 9999 

# 192.168.122.31
PEER_ZMQ_URL   = 'tcp://192.168.122.30:3333'
SERVER_ZMQ_URL = 'tcp://192.168.122.31:3333'
PARTY_ID       = 10000 

BT_PRE         = f'bt{PARTY_ID}_'

REDIS_HOST     = '127.0.0.1'
REDIS_PORT     = 6379

BT_MAX_SIZE    = 9000

g_running = True

def my_sig_int(signum, frame):
    global g_running
    g_running = False

def handle_peer(peer_socket, r_conn, ckks, nslots):
    a = np.random.randint(low=-POWER, high=POWER, size=nslots)
    a2 = np.random.randint(low=-POWER, high=POWER, size=nslots)
    a1 = a - a2

    pt_a = ckks.CreatePtxtArray(tuple(a.astype(float)))
    ct_a = ckks.Encrypt(pt_a)
    s_ct_a = ct_a.Get()

    beaver_tripple_uuid = BT_PRE + uuid.uuid1().hex

    msg = dict(cmd='HANDLE', s_ct_a=s_ct_a, a2=tuple(a2), beaver_tripple_uuid=beaver_tripple_uuid)
    peer_socket.send_pyobj(msg)

    return tuple(a1), beaver_tripple_uuid

def handle_server(r_conn, ckks, s_ct_a, a2, beaver_tripple_uuid, nslots):
    b = np.random.randint(low=-POWER, high=POWER, size=nslots)
    b1 = np.random.randint(low=-POWER, high=POWER, size=nslots)
    b2 = b - b1

    pt_b = ckks.CreatePtxtArray(tuple(b.astype(float)))

    c2 = np.random.randint(low=-PRODUCT_POWER, high=PRODUCT_POWER, size=nslots)
    pt_c2 = ckks.CreatePtxtArray(tuple(c2.astype(float)))

    ct_a = ckks.LoadCtxt(s_ct_a)

    ct_c1 = ct_a * pt_b - pt_c2

    s_ct_c1 = ct_c1.Get()

    redis_value = zlib.compress(pickle.dumps(dict(a=a2, b=tuple(b2), c=tuple(c2))))
    r_conn.set(beaver_tripple_uuid, redis_value)

    return s_ct_c1, tuple(b1)

def server(server_socket, r_conn):
    print('******** server')

    is_ready = False
    ckks = None
    nslots = 0

    poller = zmq.Poller()
    poller.register(socket=server_socket, flags=zmq.POLLIN)

    while g_running:
        events = dict(poller.poll(1000))
        if server_socket not in events:
            continue

        response = dict(status_code=1)
        msg = server_socket.recv_pyobj()
        cmd = msg.get('cmd')
        if 'HANDLE' == cmd:
            if is_ready:
                s_ct_a = msg.get('s_ct_a')
                a2 = msg.get('a2')
                beaver_tripple_uuid = msg.get('beaver_tripple_uuid')
                if s_ct_a is not None and a2 is not None and beaver_tripple_uuid is not None:
                    response['status_code'] = 0
                    response['s_ct_c1'], response['b1'] = handle_server(r_conn, ckks, s_ct_a, a2, beaver_tripple_uuid, nslots)
                else:
                    response['msg'] = 'wrong params'
            else:
                response['msg'] = 'not inited'
        elif 'INIT' == cmd:
            compressed_s_cont = msg.get('compressed_s_cont')
            s_pk = msg.get('s_pk')
            if all((compressed_s_cont, s_pk)):
                response['status_code'] = 0
                s_cont = zlib.decompress(compressed_s_cont)
                ckks = pyhelib.CKKS.LoadContext(s_cont)
                nslots = ckks.nslots
                ckks.LoadPK(s_pk)
                is_ready = True
            else:
                response['msg'] = 'wrong params'
        else:
            response['msg'] = 'wrong cmd'

        server_socket.send_pyobj(response)

def peer(peer_socket, r_conn):
    print('******** peer')

    ckks = pyhelib.CKKS()

    s_cont = ckks.GetContext()
    compressed_s_cont = zlib.compress(s_cont)

    nslots = ckks.nslots

    ckks.GenKey()
    s_pk = ckks.GetPK()

    msg_init = dict(cmd='INIT', compressed_s_cont=compressed_s_cont, s_pk=s_pk)

    peer_socket.send_pyobj(msg_init)
    
    poller = zmq.Poller()
    poller.register(socket=peer_socket, flags=zmq.POLLIN)

    inited = False
    dbsize = 0
    a1 = None

    while g_running:
        events = dict(poller.poll(1000))
        if not events:
            if inited:
                if 0 == dbsize:
                    keys = r_conn.keys(f'{BT_PRE}*')
                    dbsize = BT_MAX_SIZE - len(keys)

                if dbsize > 0 and a1 is None:
                    a1, beaver_tripple_uuid = handle_peer(peer_socket, r_conn, ckks, nslots)
            continue

        if peer_socket not in events:
            continue

        response = peer_socket.recv_pyobj()
        status_code = response.get('status_code')
        if 0 != status_code:
            break

        if inited:
            b1 = response.get('b1')
            s_ct_c1 = response.get('s_ct_c1')

            ct_c1 = ckks.LoadCtxt(s_ct_c1)
            pt_c1 = ckks.Decrypt(ct_c1)

            c1 = tuple(map(lambda x: round(x), pt_c1.GetValues()))

            redis_value = zlib.compress(pickle.dumps(dict(a=a1, b=b1, c=c1)))
            r_conn.set(beaver_tripple_uuid, redis_value)
            dbsize -= 1
            a1 = None
        else:
            inited = True

def main():
    thd_server = threading.Thread(target=server, args=(server_socket, r_conn))
    thd_server.start()

    thd_peer = threading.Thread(target=peer, args=(peer_socket, r_conn))
    thd_peer.start()

    thd_server.join()
    thd_peer.join()

    print('done')

if '__main__' == __name__:
    signal.signal(signal.SIGINT, my_sig_int)

    context = zmq.Context()
    peer_socket = context.socket(zmq.REQ)
    peer_socket.connect(PEER_ZMQ_URL)
    server_socket = context.socket(zmq.REP)
    server_socket.bind(SERVER_ZMQ_URL)

    pool = ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0, password=None)
    r_conn = StrictRedis(connection_pool=pool)

    main()

    r_conn.close()

    peer_socket.close()
    server_socket.close()

    context.destroy()
    print('ok')
