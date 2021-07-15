import socket
import tensorflow as tf
import pickle
import time
import sys

class Client(object):

    def __init__(self, port):
        self.port = port
        self.client = socket.socket()

    def connectToServer(self):
        self.client.connect(('localhost', self.port))

    def sendData(self, data):
        data = pickle.dumps(data)
        # print("send data", sys.getsizeof(data))
        self.client.send(data)

    def readData(self):
        data = self.client.recv(49152)
        # print("read data", sys.getsizeof(data))
        data = pickle.loads(data)
        # print(data)
        return data

    def closeClient(self):
        self.client.close()

    def getLoss(self):
        return self.loss

    def setLoss(self, loss):
        self.loss = loss


if __name__ == "__main__":
    client = Client(8888)
    client.connectToServer()
    z = client.readData()
    print(z)
    z = tf.convert_to_tensor(z)
    print(z)
    time.sleep(1)
    loss = 0.8888
    client.sendData(loss)