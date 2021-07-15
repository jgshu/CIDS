import socket
from utils.MyThread import MyThread
import tensorflow as tf
import time
import pickle
import sys
import bigan.kdd_utilities as network
import numpy as np

class Server(object):

    def __init__(self, port):
        self.port = port
        self.server = socket.socket()
        self.server.bind(('localhost', port))
        self.server.listen(5)
        self.conn = None
        self.addr = None
        self.loss = -1000.0

    def connectToClient(self):
        print("start to accept client")
        self.conn, self.addr = self.server.accept()
        print("sucess connect to {}-{}", self.conn, self.addr)
        # while True:
        #     data = self.conn.recv(10)
        #     self.loss = data
        #     print("received loss{}", self.loss)

    def sendData(self, data):
        data = pickle.dumps(data)
        # print("send data", sys.getsizeof(data))
        self.conn.send(data)

    def readData(self):
        data = self.conn.recv(49152)
        # print("read data", sys.getsizeof(data))
        data = pickle.loads(data)
        # print("read data:", data)
        return data

    def closeServer(self):
        self.server.close()


    def getLoss(self):
        return self.loss

    def setLoss(self, loss):
        self.loss = loss


if __name__ == "__main__":
    sess = tf.Session()
    # gen = network.decoder
    # z = tf.random_normal([50, 32])
    # x_gen = gen(z, is_training=False)
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # print(x_gen)
    # x_gen = x_gen.eval(session=sess)
    # print(x_gen)
    # x_gen = pickle.dumps(x_gen)
    # print(x_gen)
    # print(sys.getsizeof(x_gen))
    server = Server(8888)
    server.connectToClient()
    z = tf.random_normal([50, 32])
    print(z)
    z = z.eval(session=sess)
    server.sendData(z)
    t1 = MyThread(server)
    t1.start()
    t1.join()
    print("loss:", server.getLoss())
