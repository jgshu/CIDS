import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import bigan.kdd_utilities as network
import data.kdd_half as data
from sklearn.metrics import precision_recall_fscore_support
from socket_communication.server import Server
from utils.MyThread import MyThread
import time
RANDOM_SEED = 13
FREQ_PRINT = 20 # print frequency image tensorboard [20]

server1 = None
server2 = None


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var

    return ema_getter

def display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree):
    '''See parameters
    '''
    print('Batch size: ', batch_size)
    print('Starting learning rate: ', starting_lr)
    print('EMA Decay: ', ema_decay)
    print('Weight: ', weight)
    print('Method for discriminator: ', method)
    print('Degree for L norms: ', degree)

def display_progression_epoch(j, id_max):
    '''See epoch progression
    '''
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def create_logdir(method, weight, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "bigan/train_logs/kdd/{}/{}/{}/central".format(weight, method, rd)

def train_and_test(nb_epochs, weight, method, degree, random_seed):
    """ Runs the Bigan on the KDD dataset

    Note:
        Saves summaries on tensorboard. To display them, please use cmd line
        tensorboard --logdir=model.training_logdir() --port=number
    Args:
        nb_epochs (int): number of epochs
        weight (float, optional): weight for the anomaly score composition
        method (str, optional): 'fm' for ``Feature Matching`` or "cross-e"
                                     for ``cross entropy``, "efm" etc.
        anomalous_label (int): int in range 0 to 10, is the class/digit
                                which is considered outlier
    """
    logger = logging.getLogger("BiGAN.train.kdd.{}".format(method))

    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    trainx, trainy = data.get_train(1)
    trainx_copy = trainx.copy()
    testx, testy = data.get_test(1)

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    logger.info('Building training graph...')

    logger.warn("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = network.decoder

    with tf.variable_scope('generator_model'):
        z = tf.random_normal([batch_size, latent_dim])
        z_1 = tf.placeholder(tf.float32, shape=(batch_size, latent_dim), name="z_1")
        x_gen1 = gen(z_1, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
        # z_2 = tf.placeholder(tf.float32, shape=(batch_size, latent_dim), name="z_2")
        # x_gen2 = gen(z_2, is_training=is_training_pl, reuse=tf.AUTO_REUSE)

    with tf.name_scope('loss_functions'):
        g1 = tf.placeholder(tf.float32, shape=(50, 121), name="g1")
        g2 = tf.placeholder(tf.float32, shape=(50, 121), name="g2")
        total = 0
        for i in range(50):
            v_g = tf.slice(g1, [i, 0], [1, 121])
            v_x_gen = tf.slice(x_gen1, [i, 0], [1, 121])
            v_g = tf.reshape(v_g, [121, 1])
            temp = tf.matmul(v_x_gen, v_g)
            total += temp
        for i in range(50):
            v_g = tf.slice(g2, [i, 0], [1, 121])
            v_x_gen = tf.slice(x_gen1, [i, 0], [1, 121])
            v_g = tf.reshape(v_g, [121, 1])
            temp = tf.matmul(v_x_gen, v_g)
            total += temp
        loss_generator = total/100

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        gvars = [var for var in tvars if 'generator_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)




    # logdir = create_logdir(weight, method, random_seed)
    # variables = tf.contrib.framework.get_variables_to_restore()
    # restore_variable_list2 = [v for v in variables if 'generator_model' in v.name]
    # for i in restore_variable_list2:
    #     print(i)
    # exit()




    logger.info('Start training...')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    saver = tf.train.Saver(max_to_keep=1000, keep_checkpoint_every_n_hours=2)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        logger.info('Initialization done')
        train_batch = 0
        epoch = 0
        a1 = 1/3
        a2 = 1/3
        a3 = 1/3
        while epoch < nb_epochs:

            lr = starting_lr
            begin = time.time()

             # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]

            # training
            for t in range(nr_batches_train):
                
                display_progression_epoch(t, nr_batches_train)             
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                z1 = sess.run(z)
                # z2 = sess.run(z)

                feed_dict = {is_training_pl: True,
                             z_1: z1}
                x_gen_1 = sess.run(x_gen1, feed_dict=feed_dict)
                # feed_dict = {is_training_pl: True,
                #              z_2: z2}
                # x_gen_2 = sess.run(x_gen2, feed_dict=feed_dict)


                # send the random vector and generated flow to the distributed SDN
                server1.sendData(z1)
                server1.readData()
                server2.sendData(z1)
                # receive ack
                server2.readData()

                server1.sendData(x_gen_1)
                server1.readData()
                # read ack
                server2.sendData(x_gen_1)
                server2.readData()

                # wait to recv loss from the distributed SDN
                t1 = MyThread(server1)
                t2 = MyThread(server2)
                t1.start()
                t2.start()
                t1.join()
                t2.join()

                loss1 = server1.getLoss()
                loss2 = server2.getLoss()
                loss1 = np.array(loss1)
                loss1 = loss1.reshape(50, 121)
                loss2 = np.array(loss2)
                loss2 = loss2.reshape(50, 121)

                # train generator and encoder
                feed_dict = {is_training_pl: True,
                             g1: loss1,
                             g2: loss2,
                             z_1: z1,
                             # z_2: z2,
                             learning_rate: lr}
                _ = sess.run([train_gen_op],
                                           feed_dict=feed_dict)

                train_batch += 1


            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds  "
                  % (epoch, time.time() - begin))

            epoch += 1


            save_path = saver.save(sess, 'central/central_{}.ckpt'.format(epoch))
            print("model save to", save_path)


def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        global server1
        global server2
        # 开启socket server连接
        server1 = Server(8888)
        server1.connectToClient()
        server2 = Server(8889)
        server2.connectToClient()
        train_and_test(nb_epochs, weight, method, degree, random_seed)