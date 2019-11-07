from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket
import threading

import deeptensor as dt


BUF_SIZE=4096

class DataPacket(object):
    def __init__(self, data, data_type=0):
        self._data = data
        self._data_type = data_type
        self._prefix = b'\xBE\xEF'
        self._append = b'\xCA\xFE'

    def pack(self):
        self._data_len = int(len(self._data))
        packet = bytearray(self._prefix)
        packet += self._data_type.to_bytes(1, byteorder = 'little')
        packet += self._data_len.to_bytes(4, byteorder = 'little')
        packet += self._data
        packet += self._append
        return bytes(packet)

    def unpack(self, bytes_data):
        cur = 0
        buf_len = len(bytes_data)
        while cur < buf_len:
            if bytes_data[cur] == 0xBE and bytes_data[cur+1] == 0xEF:
                data_type = int.from_bytes(bytes_data[cur+2:cur+2+1], byteorder='little')
                data_len = int.from_bytes(bytes_data[cur+2+1:cur+2+1+4], byteorder='little')
                data_start = cur+2+1+4
                data_end = data_start+data_len
                if (data_end+2) <= buf_len:
                    if bytes_data[data_end] == 0xCA and bytes_data[data_end+1] == 0xFE:
                        self._data = bytes_data[data_start:data_end]
                        self._data_type = data_type
                        self._data_len = data_len
                        cur = data_end+2
                        break
                else:
                    break
            cur += 1
        return cur


class DataLink(object):

    def __init__(self, name="datalink", host=None, port=6001):
        self._name = name
        self._host = host
        self._port = port

        self._running = True
        self._connected = False

        self._server = None
        self._clients = {}
        self._client = None

        self._recv_fns = set()

        self.init()

    def handle_client_connection(self, client_id, client_socket):
        recv_bytes = bytearray(b'')
        while self._running:
            try:
                recv_data = client_socket.recv(BUF_SIZE)
                if len(recv_data) == 0:
                    print('\n[Server] Client disconnected, {}'.format(client_id))
                    client_socket.close()
                    self._clients.pop(client_id, None)
                    break
                else:
                    recv_bytes += recv_data
                    while True:
                        packet = DataPacket(None)
                        consumed = packet.unpack(recv_bytes)
                        if packet._data is not None:
                            for fn in self._recv_fns:
                                fn(client_socket, packet)
                        if consumed == 0:
                            break
                        else:
                            recv_bytes = recv_bytes[consumed:]
            except:
                self._clients.pop(client_id, None)

    def server_listen(self):
        while self._running:
            try:
                client_sock, address = self._server.accept()
                client_id = "{}:{}".format(address[0], address[1])
                print('\n[Server] Accepted connection from {}'.format(client_id))
                client_handler = threading.Thread(
                    target=self.handle_client_connection,
                    args=(client_id, client_sock,)
                )
                client_handler.start()
                self._clients[client_id] = client_sock
            except:
                print('\n[Server] Cannot accept connection. Shutting down.')
                pass

    def client_listen(self):
        recv_bytes = bytearray(b'')
        while self._running:
            try:
                recv_data = self._client.recv(BUF_SIZE)
                if len(recv_data) == 0:
                    print('[Client] Server disconnected')
                    self._client.close()
                    break
                else:
                    recv_bytes += recv_data
                    while True:
                        packet = DataPacket(None)
                        consumed = packet.unpack(recv_bytes)
                        if packet._data is not None:
                            #print(packet._data)
                            for fn in self._recv_fns:
                                fn(self._client, packet)
                        if consumed == 0:
                            break
                        else:
                            recv_bytes = recv_bytes[consumed:]
            except:
                self._client.close()
                self._connected = False

    def init(self):
        if self._host is None:
            # Server
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind(('0.0.0.0', self._port))
            self._server.listen(4)

            server_handler = threading.Thread(
                target=self.server_listen,
            )
            server_handler.start()
            self._connected = True

        else:
            # Client
            self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._client.connect((self._host, self._port))

            client_handler = threading.Thread(
                target=self.client_listen,
            )
            client_handler.start()
            self._connected = True

    def send(self, str_data):
        if not self._connected:
            return False

        try:
            if self._host is None:
                for k, c in self._clients.items():
                    c.send(DataPacket(str_data.encode()).pack())
            else:
                self._client.send(DataPacket(str_data.encode()).pack())
        except:
            return False

        return True

    def send_sock(self, socket, str_data):
        if not self._connected:
            return False

        try:
            socket.send(DataPacket(str_data.encode()).pack())
        except:
            return False

        return True

    def send_opt(self, opt):
        return self.send(opt.dumps())

    def send_opt_sock(self, socket, opt):
        return self.send_sock(socket, opt.dumps())

    def register_recv(self, callback):
        self._recv_fns.add(callback)

    def close(self):
        if not self._connected:
            return
        self._connected = False
        self._running = False

        try:
            if self._host is None:
                for k, c in self._clients.items():
                    c.close()
                self._server.shutdown(socket.SHUT_RDWR)
                self._server.close()
            else:
                self._client.close()
        except:
            return


_datalink = None

def datalink_start(host=None, port=6001):
    global _datalink
    _datalink = DataLink(name='datalink', host=host, port=port)

def datalink_close():
    global _datalink
    if _datalink is not None:
        _datalink.close()
        _datalink = None

def datalink():
    global _datalink
    return _datalink

def datalink_register_recv(recv_fn):
    global _datalink
    if _datalink is not None:
        _datalink.register_recv(recv_fn)

def datalink_send_opt(opt):
    global _datalink
    if _datalink is not None:
        _datalink.send_opt(opt)
