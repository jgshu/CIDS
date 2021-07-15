import threading


class MyThread(threading.Thread):
    def __init__(self, server):
        super(MyThread, self).__init__()
        self.server = server

    def run(self):
        loss = self.server.readData()
        self.server.setLoss(loss)
        # send ack
        self.server.sendData(1)