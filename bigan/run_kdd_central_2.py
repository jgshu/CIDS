import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import bigan.kdd_utilities as network
import data.kdd_half as data
import data.kdd as data_
from sklearn.metrics import precision_recall_fscore_support
from socket_communication.server import Server
from utils.MyThread import MyThread
import time
import random
RANDOM_SEED = 13
FREQ_PRINT = 20 # print frequency image tensorboard [20]
import six
from six.moves import zip

server1 = None
server2 = None
server3 = None

tf_vars = None
placeholders = None
assign_ops = None

i = 0


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

def averageModels(model1, model2, output, w1, w2):
    global tf_vars
    global placeholders
    global assign_ops
    global i
    print(model1)
    print(i)
    var_list = tf.train.list_variables(model1)
    # print(var_list)
    var_values, var_dtypes = {}, {}
    for (name, shape) in var_list:
        var_values[name] = np.zeros(shape)
    reader1 = tf.train.load_checkpoint(model1)
    reader2 = tf.train.load_checkpoint(model2)
    for name in var_values:
        tensor1 = reader1.get_tensor(name)
        tensor2 = reader2.get_tensor(name)
        var_dtypes[name] = tensor1.dtype
        var_values[name] = w1*tensor1 + w2*tensor2
    if i == 0:
        i = i + 1
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            tf_vars = [
                tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
                for v in var_values
            ]
        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    variables = tf.all_variables()
    restore_variable_list1 = [v for v in variables if ('encoder_model' in v.name) or ('generator_model' in v.name)]
    saver = tf.train.Saver(restore_variable_list1, max_to_keep=1000, keep_checkpoint_every_n_hours=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for p, assign_ops, (name, value) in zip(placeholders, assign_ops, six.iteritems(var_values)):
            sess.run(assign_ops, {p: value})
        saver.save(sess, output)
    # var_list_2 = tf.train.list_variables('model3.ckpt')
    # print(var_list)
    # print(var_list_2)

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
    # logger = logging.getLogger("BiGAN.train.kdd.{}".format(method))

    # Placeholders
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    trainx, trainy = data.get_train(1)
    trainx_copy = trainx.copy()
    # testx, testy = data.get_test(1)

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    # nr_batches_test = int(testx.shape[0] / batch_size)

    # logger.info('Building training graph...')
    #
    # logger.warn("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    dis = network.discriminator

    with tf.variable_scope('discriminator_model'):
        z_gen_1 = tf.placeholder(tf.float32, shape=(50, 32), name="z_gen_1")
        input_1 = tf.placeholder(tf.float32, shape=(50, 121), name="input_1")
        l_encoder1, inter_layer_inp1 = dis(z_gen_1, input_1, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
        z_1 = tf.placeholder(tf.float32, shape=(50, 32), name="z_1")
        x_gen_1 = tf.placeholder(tf.float32, shape=(50, 121), name="x_gen_1")
        l_generator1, inter_layer_rct1 = dis(z_1, x_gen_1, is_training=is_training_pl, reuse=True)

        z_gen_2 = tf.placeholder(tf.float32, shape=(50, 32), name="z_gen_2")
        input_2 = tf.placeholder(tf.float32, shape=(50, 121), name="input_2")
        l_encoder2, inter_layer_inp2 = dis(z_gen_2, input_2, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
        z_2 = tf.placeholder(tf.float32, shape=(50, 32), name="z_2")
        x_gen_2 = tf.placeholder(tf.float32, shape=(50, 121), name="x_gen_2")
        l_generator2, inter_layer_rct2 = dis(z_2, x_gen_2, is_training=is_training_pl, reuse=True)

        z_gen_3 = tf.placeholder(tf.float32, shape=(50, 32), name="z_gen_3")
        input_3 = tf.placeholder(tf.float32, shape=(50, 121), name="input_3")
        l_encoder3, inter_layer_inp3 = dis(z_gen_3, input_3, is_training=is_training_pl, reuse=tf.AUTO_REUSE)
        z_3 = tf.placeholder(tf.float32, shape=(50, 32), name="z_3")
        x_gen_3 = tf.placeholder(tf.float32, shape=(50, 121), name="x_gen_3")
        l_generator3, inter_layer_rct3 = dis(z_3, x_gen_3, is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        # discriminator
        loss_dis_enc1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder1),logits=l_encoder1))
        loss_dis_gen1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator1),logits=l_generator1))
        loss_discriminator1 = loss_dis_gen1 + loss_dis_enc1
        # compute the loss function of G and E
        loss_generator1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator1), logits=l_generator1))
        loss_encoder1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder1), logits=l_encoder1))

        #compute the gradient of grnerator and encoder
        cross_g1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator1), logits=l_generator1)
        cross_e1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder1), logits=l_encoder1)
        x_g1 = tf.gradients(cross_g1, x_gen_1)
        x_e1 = tf.gradients(cross_e1, z_gen_1)

        # discriminator
        loss_dis_enc2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder2),logits=l_encoder2))
        loss_dis_gen2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator2),logits=l_generator2))
        loss_discriminator2 = loss_dis_gen2 + loss_dis_enc2
        # compute the loss function of G and E
        loss_generator2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator2), logits=l_generator2))
        loss_encoder2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder2), logits=l_encoder2))

        #compute the gradient of grnerator and encoder
        cross_g2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator2), logits=l_generator2)
        cross_e2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder2), logits=l_encoder2)
        x_g2 = tf.gradients(cross_g2, x_gen_2)
        x_e2 = tf.gradients(cross_e2, z_gen_2)

        # discriminator
        loss_dis_enc3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder3),logits=l_encoder3))
        loss_dis_gen3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator3),logits=l_generator3))
        loss_discriminator3 = loss_dis_gen3 + loss_dis_enc3
        # compute the loss function of G and E
        loss_generator3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator3), logits=l_generator3))
        loss_encoder3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder3), logits=l_encoder3))

        #compute the gradient of grnerator and encoder
        cross_g3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator3), logits=l_generator3)
        cross_e3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder3), logits=l_encoder3)
        x_g3 = tf.gradients(cross_g3, x_gen_3)
        x_e3 = tf.gradients(cross_e3, z_gen_3)

        loss_discriminator = (loss_discriminator1+loss_discriminator2+loss_discriminator3)/3

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')


        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for estimation
        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

    # logdir = create_logdir(weight, method, random_seed)
    # variables = tf.contrib.framework.get_variables_to_restore()
    # restore_variable_list2 = [v for v in variables if 'generator_model' in v.name]
    # for i in restore_variable_list2:
    #     print(i)
    # exit()

    # logger.info('Start training...')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    saver = tf.train.Saver(max_to_keep=1000, keep_checkpoint_every_n_hours=2)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # logger.info('Initialization done')
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
            train_loss_dis1, train_loss_gen1, train_loss_enc1 = [0, 0, 0]
            train_loss_dis2, train_loss_gen2, train_loss_enc2 = [0, 0, 0]
            i = 0
            # training
            for t in range(nr_batches_train):

                display_progression_epoch(t, nr_batches_train)
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                server1.sendData("1")
                z1_1 = server1.readData()
                z1_1 = np.array(z1_1)
                server1.sendData("1")
                _x_gen_1 = server1.readData()
                _x_gen_1 = np.array(_x_gen_1)
                server1.sendData("1")
                data1 = server1.readData()
                data1 = np.array(data1)
                server1.sendData("1")
                _z_gen_1 = server1.readData()
                _z_gen_1 = np.array(_z_gen_1)
                server1.sendData("1")
                # 发送信号帧使得client2开始发送数据
                server2.sendData("1")
                z1_2 = server2.readData()
                z1_2 = np.array(z1_2)
                server2.sendData("1")
                _x_gen_2 = server2.readData()
                _x_gen_2 = np.array(_x_gen_2)
                server2.sendData("1")
                data2 = server2.readData()
                data2 = np.array(data2)
                server2.sendData("1")
                _z_gen_2 = server2.readData()
                _z_gen_2 = np.array(_z_gen_2)
                server2.sendData("1")
                # 发送信号帧使得client3开始发送数据
                server3.sendData("1")
                z1_3 = server3.readData()
                z1_3 = np.array(z1_3)
                server3.sendData("1")
                _x_gen_3 = server3.readData()
                _x_gen_3 = np.array(_x_gen_3)
                server3.sendData("1")
                data3 = server3.readData()
                data3 = np.array(data3)
                server3.sendData("1")
                _z_gen_3 = server3.readData()
                _z_gen_3 = np.array(_z_gen_3)
                server3.sendData("1")


                # feed_dict = {input_1: data1,
                #              z_gen_1: _z_gen_1,
                #              z_1: z1_1,
                #              x_gen_1: _x_gen_1,
                #              is_training_pl: True,
                #              learning_rate: lr}

                # _, ld, lg1, le1, x_g_1, x_e_1 = sess.run([train_dis_op,
                #                                         loss_discriminator,
                #                                         loss_generator1,
                #                                         loss_encoder1,
                #                                         x_g1,
                #                                         x_e1,
                #
                #                                           ],
                #                                          feed_dict=feed_dict)
                # train_loss_dis1 += ld
                # train_loss_gen1 += lg1
                # train_loss_enc1 += le1


                # train discriminator
                feed_dict = {input_1: data1,
                             z_gen_1: _z_gen_1,
                             z_1: z1_1,
                             x_gen_1: _x_gen_1,
                             input_2: data2,
                             z_gen_2: _z_gen_2,
                             z_2: z1_2,
                             x_gen_2: _x_gen_2,
                             input_3: data3,
                             z_gen_3: _z_gen_3,
                             z_3: z1_3,
                             x_gen_3: _x_gen_3,
                             is_training_pl: True,
                             learning_rate: lr}

                _, ld, lg1, le1, x_g_1, x_e_1, lg2, le2, x_g_2, x_e_2, lg3, le3, x_g_3, x_e_3 = sess.run([train_dis_op,
                                                        loss_discriminator,
                                                        loss_generator1,
                                                        loss_encoder1,
                                                        x_g1,
                                                        x_e1,
                                                        loss_generator2,
                                                        loss_encoder2,
                                                        x_g2,
                                                        x_e2,
                                                        loss_generator3,
                                                        loss_encoder3,
                                                        x_g3,
                                                        x_e3
                                                          ],
                                                         feed_dict=feed_dict)
                train_loss_dis1 += ld
                train_loss_gen1 += lg1
                train_loss_enc1 += le1



                server1.sendData(x_g_1)
                server1.readData()
                server1.sendData(x_e_1)
                server1.readData()
                # print("datasize:",sys.getsizeof(x_g_1)+sys.getsizeof(x_e_1))
                # exit()
                server2.sendData(x_g_2)
                server2.readData()
                server2.sendData(x_e_2)
                server2.readData()

                server3.sendData(x_g_3)
                server3.readData()
                server3.sendData(x_e_3)
                server3.readData()

                train_loss_dis2 += ld
                train_loss_gen2 += lg2
                train_loss_enc2 += le2

                train_batch += 1


            train_loss_gen1 /= nr_batches_train
            train_loss_enc1 /= nr_batches_train
            train_loss_dis1 /= nr_batches_train

            train_loss_gen2 /= nr_batches_train
            train_loss_enc2 /= nr_batches_train
            train_loss_dis2 /= nr_batches_train



            # logger.info('Epoch terminated')
            print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen1, train_loss_enc1, train_loss_dis1))
            print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen2, train_loss_enc2, train_loss_dis2))


            epoch += 1
            if epoch % 5 == 0:
                save_path = saver.save(sess, 'central/central_{}.ckpt'.format(epoch))
                print("model save to", save_path)
            # if epoch % 5 == 0:
            #     items = [1, 2, 3]
            #     random.shuffle(items)
            #     # while items[0] == 1 and items[1] == 2 and items[2] == 3:
            #     #     random.shuffle(items)
            #     server1.sendData(items[0])
            #     server1.readData()
            #     server2.sendData(items[1])
            #     server2.readData()
            #     server3.sendData(items[2])
            #     server3.readData()
                # if epoch % 5 == 0:
                #     time.sleep(2)
                #     start = time.clock()
                #     i = random.randint(1, 3)
                #     time.sleep(0.5)
                #     j = random.randint(1, 3)
                #     time.sleep(0.5)
                #     k = random.randint(1, 3)
                #     averageModels('distributed/1/distributed_{}.ckpt'.format(epoch),
                #                   'distributed/1/distributed_{}.ckpt'.format(epoch),
                #                   'distributed/4/distributed_{}.ckpt'.format(epoch), 0.25, 0.25)
                #     averageModels('distributed/2/distributed_{}.ckpt'.format(epoch),
                #                   'distributed/2/distributed_{}.ckpt'.format(epoch),
                #                   'distributed/5/distributed_{}.ckpt'.format(epoch), 0.25, 0.25)
                #     averageModels('distributed/3/distributed_{}.ckpt'.format(epoch),
                #                   'distributed/3/distributed_{}.ckpt'.format(epoch),
                #                   'distributed/6/distributed_{}.ckpt'.format(epoch), 0.25, 0.25)
                #     a1 = a1/2
                #     a2 = a2/2
                #     a3 = a3/2
                #     temp1 = a1
                #     temp2 = a2
                #     temp3 = a3
                #     if i == 1:
                #         w1 = a1/(a1+a1)
                #         w2 = a1/(a1+a1)
                #         averageModels('distributed/4/distributed_{}.ckpt'.format(epoch), 'distributed/4/distributed_{}.ckpt'.format(epoch), 'distributed/7/distributed_{}.ckpt'.format(epoch), w1, w2)
                #         a1 = a1 + a1
                #     elif i == 2:
                #         w1 = a2/(temp1+a2)
                #         w2 = temp1/(temp1+a2)
                #         averageModels('distributed/5/distributed_{}.ckpt'.format(epoch), 'distributed/4/distributed_{}.ckpt'.format(epoch), 'distributed/8/distributed_{}.ckpt'.format(epoch), w1, w2)
                #         a2 = a2 + temp1
                #     elif i == 3:
                #         w1 = a3/(temp1+a3)
                #         w2 = temp1/(temp1+a3)
                #         averageModels('distributed/6/distributed_{}.ckpt'.format(epoch), 'distributed/4/distributed_{}.ckpt'.format(epoch), 'distributed/9/distributed_{}.ckpt'.format(epoch), w1, w2)
                #         a3 = a3 + temp1
                #     if j == 1:
                #         w1 = a1/(temp2+a1)
                #         w2 = temp2/(temp2+a1)
                #         averageModels('distributed/4/distributed_{}.ckpt'.format(epoch), 'distributed/5/distributed_{}.ckpt'.format(epoch), 'distributed/7/distributed_{}.ckpt'.format(epoch), w1, w2)
                #         a1 = a1 + temp2
                #     elif j == 2:
                #         w1 = a2/(a2+a2)
                #         w2 = a2/(a2+a2)
                #         averageModels('distributed/5/distributed_{}.ckpt'.format(epoch), 'distributed/5/distributed_{}.ckpt'.format(epoch), 'distributed/8/distributed_{}.ckpt'.format(epoch), w1, w2)
                #         a2 = a2 + a2
                #     elif j == 3:
                #         w1 = a3/(temp2+a3)
                #         w2 = temp2/(temp2+a3)
                #         averageModels('distributed/6/distributed_{}.ckpt'.format(epoch), 'distributed/5/distributed_{}.ckpt'.format(epoch), 'distributed/9/distributed_{}.ckpt'.format(epoch), w1, w2)
                #         a3 = a3 + temp2
                #     if k == 1:
                #         w1 = a1/(temp3+a1)
                #         w2 = temp3/(temp3+a1)
                #         averageModels('distributed/4/distributed_{}.ckpt'.format(epoch), 'distributed/3/distributed_{}.ckpt'.format(epoch), 'distributed/7/distributed_{}.ckpt'.format(epoch), w1, w2)
                #         a1 = a1 + temp3
                #     elif k == 2:
                #         w1 = a2/(temp3+a2)
                #         w2 = temp3/(temp3+a2)
                #         averageModels('distributed/5/distributed_{}.ckpt'.format(epoch), 'distributed/3/distributed_{}.ckpt'.format(epoch), 'distributed/8/distributed_{}.ckpt'.format(epoch), w1, w2)
                #         a2 = a2 + temp3
                #     elif k == 3:
                #         w1 = a3/(a3+a3)
                #         w2 = a3/(a3+a3)
                #         averageModels('distributed/6/distributed_{}.ckpt'.format(epoch), 'distributed/3/distributed_{}.ckpt'.format(epoch), 'distributed/9/distributed_{}.ckpt'.format(epoch), w1, w2)
                #         a3 = a3 + a3
                #     if i!=1 and j!=1 and k!= 1:
                #         averageModels('distributed/4/distributed_{}.ckpt'.format(epoch),
                #                       'distributed/4/distributed_{}.ckpt'.format(epoch),
                #                       'distributed/7/distributed_{}.ckpt'.format(epoch), 0.5, 0.5)
                #     if i!=2 and j!=2 and k!= 2:
                #         averageModels('distributed/5/distributed_{}.ckpt'.format(epoch),
                #                       'distributed/5/distributed_{}.ckpt'.format(epoch),
                #                       'distributed/8/distributed_{}.ckpt'.format(epoch), 0.5, 0.5)
                #     if i!=3 and j!=3 and k!= 3:
                #         averageModels('distributed/6/distributed_{}.ckpt'.format(epoch),
                #                       'distributed/6/distributed_{}.ckpt'.format(epoch),
                #                       'distributed/9/distributed_{}.ckpt'.format(epoch), 0.5, 0.5)
                #     end = time.clock()
                #     print("i", i)
                #     print("j", j)
                #     print("k", k)
                #     print("a1:", a1)
                #     print("a2:", a2)
                #     print("a3:", a3)
                #     print("save model finished, time:{}".format(end-start))
                #     server1.sendData("1")
                #     server2.sendData("2")
                #     server3.sendData("3")











def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        global server1
        global server2
        global server3
        # 开启socket server连接
        server1 = Server(8888)
        server1.connectToClient()
        server2 = Server(8889)
        server2.connectToClient()
        server3 = Server(8890)
        server3.connectToClient()
        train_and_test(nb_epochs, weight, method, degree, random_seed)