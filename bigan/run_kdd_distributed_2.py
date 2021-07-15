import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import bigan.kdd_utilities as network
import data.kdd_half as data
import random
import data.kdd as data_
from sklearn.metrics import precision_recall_fscore_support
from socket_communication.client import Client
import time

RANDOM_SEED = 13
FREQ_PRINT = 20 # print frequency image tensorboard [20]
client = None # socket connect to the central SDN


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

def create_logdir(method, weight, rd, id):
    """ Directory to save training logs, weights, biases, etc."""
    return "bigan/train_logs/kdd/{}/{}/{}/distributed/{}".format(weight, method, rd, id)

def averageModels(model1, model2, output, w1, w2):
    var_list = tf.train.list_variables(model1)
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
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        tf_vars = [
            tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
            for v in var_values
        ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for p, assign_ops, (name, value) in zip(placeholders, assign_ops, six.iteritems(var_values)):
            sess.run(assign_ops, {p: value})
        saver.save(sess, 'model3.ckpt')
    # var_list_2 = tf.train.list_variables('model3.ckpt')
    # print(var_list)
    # print(var_list_2)


def train_and_test(nb_epochs, weight, method, degree, random_seed, id):
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
    trainx, trainy = data.get_train(id)
    trainx_copy = trainx.copy()
    # testx, testy = data.get_test(id)

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    nr_batches_train = int(trainx.shape[0] / batch_size)
    print(nr_batches_train)
    # if id == 2:
    #     nr_batches_train -= 3
    # nr_batches_test = int(testx.shape[0] / batch_size)

    logger.info('Building training graph...')

    logger.warn("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = network.decoder
    enc = network.encoder


    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)

    with tf.variable_scope('generator_model'):
        z1 = tf.random_normal([batch_size, latent_dim])
        z = tf.placeholder(tf.float32, shape=(batch_size, latent_dim), name="z")
        x_gen = gen(z, is_training=is_training_pl)

    with tf.name_scope('loss_functions'):
        # update the parameter of generator and encoder
        y_g = tf.placeholder(tf.float32, shape=(50, 121), name="y_g")
        y_e = tf.placeholder(tf.float32, shape=(50, 32), name='y_e')
        l_g = 0.0
        for i in range(50):
            v_g = tf.slice(y_g, [i, 0], [1, 121])
            v_x_gen = tf.slice(x_gen, [i, 0], [1, 121])
            v_g = tf.reshape(v_g, [121, 1])
            temp = tf.matmul(v_x_gen, v_g)
            l_g += temp
        loss_g = l_g / 50
        l_e = 0.0
        for i in range(50):
            v_e = tf.slice(y_e, [i, 0], [1, 32])
            v_z_gen = tf.slice(z_gen, [i, 0], [1, 32])
            v_e = tf.reshape(v_e, [32, 1])
            temp = tf.matmul(v_z_gen, v_e)
            l_e += temp
        loss_e = l_e / 50

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_g, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_e, var_list=evars)



        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)

        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)


    logdir = create_logdir(weight, method, random_seed, id)
    # variables = tf.contrib.framework.get_variables_to_restore()
    # restore_variable_list = [v for v in variables if ('generator_model' in v.name)]
    variables = tf.contrib.framework.get_variables_to_restore()
    restore_variable_list1 = [v for v in variables if ('encoder_model' in v.name) or ('generator_model' in v.name)]
    saver = tf.train.Saver(restore_variable_list1, max_to_keep=1000, keep_checkpoint_every_n_hours=2)
    restore_variable_list2 = [v for v in variables if 'encoder_model' in v.name]
    restore_variable_list3 = [v for v in variables if 'generator_model' in v.name]
    saver2 = tf.train.Saver(restore_variable_list2)
    saver3 = tf.train.Saver(restore_variable_list3)
    logger.info('Start training...')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init = tf.global_variables_initializer()  # 初始化所在的位置至关重要，以本程序为例，使用adam优化器时，会主动创建变量。
        # 因此，如果这时的初始化位置在创建adam优化器之前，则adam中包含的变量会未初始化，然后报错。本行初始化时，可以看到Adam
        # 已经声明，古不会出错
        sess.run(init)
        logger.info('Initialization done')
        writer = tf.summary.FileWriter(logdir, sess.graph)
        train_batch = 0
        epoch = 0

        while epoch < nb_epochs:

            lr = starting_lr
            begin = time.time()

             # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling dataset
            trainx_copy = trainx_copy[rng.permutation(trainx.shape[0])]
            train_loss_dis, train_loss_gen, train_loss_enc = [0, 0, 0]

            # training
            for t in range(nr_batches_train):
                
                display_progression_epoch(t, nr_batches_train)             
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # (Gz,z)
                z1_ = sess.run(z1)
                feed_dict = {is_training_pl: True,
                             z: z1_}
                _x_gen_ = sess.run([x_gen], feed_dict=feed_dict)
                _x_gen_ = np.array(_x_gen_)
                _x_gen_ = _x_gen_.reshape((50, 121))
                #(x,Ex)
                feed_dict = {is_training_pl: True,
                             input_pl: trainx[ran_from:ran_to]}
                _z_gen_ = sess.run([z_gen],feed_dict=feed_dict)
                _z_gen_ = np.array(_z_gen_)
                _z_gen_ = _z_gen_.reshape((50, 32))

                # 接受服务器开始发送数据请求
                client.readData()
                client.sendData(z1_)
                client.readData()
                client.sendData(_x_gen_)
                client.readData()
                client.sendData(trainx[ran_from:ran_to])
                client.readData()
                client.sendData(_z_gen_)
                client.readData()
                # print("datasize:---",sys.getsizeof(z1_)+sys.getsizeof(_x_gen_)+sys.getsizeof(trainx[ran_from:ran_to])+sys.getsizeof(_z_gen_))

                x_g_ = client.readData()
                # send ack
                client.sendData(1)
                x_e_ = client.readData()
                # send ack
                client.sendData(1)
                x_g_ = np.array(x_g_)
                x_g_ = x_g_.reshape((50, 121))
                x_e_ = np.array(x_e_)
                x_e_ = x_e_.reshape((50, 32))

                # train the generator and encoder
                feed_dict = {input_pl: trainx[ran_from:ran_to],
                             y_e: x_e_,
                             z: z1_,
                             y_g: x_g_,
                             is_training_pl: True,
                             learning_rate: lr}
                _, _ = sess.run([train_gen_op,
                                 train_enc_op,
                                 ],
                                feed_dict=feed_dict)

                # writer.add_summary(sm, train_batch)
                train_batch += 1


            logger.info('Epoch terminated')
            print("Epoch %d | time = %ds  "
                  % (epoch, time.time() - begin))

            epoch += 1
            if epoch % 5 == 0:
                save_path = saver.save(sess, 'distributed/{}/distributed_{}.ckpt'.format(id, epoch))
                print("model save to:", save_path)
            # if epoch % 5 == 0:
            #     i = client.readData()
            #     client.sendData("1")
            #     print("restore from the model{}".format(i))
            #     saver2.restore(sess, 'distributed/{}/distributed_{}.ckpt'.format(i, epoch))
            #     saver3.restore(sess, 'distributed/{}/distributed_{}.ckpt'.format(i, epoch))
            #     print("restore finished")
            if epoch % 150 == 0:
                # time.sleep(1)
                flag = random.randint(1, 3)
                saver2.restore(sess, 'distributed/{}/distributed_{}.ckpt'.format(flag, epoch))
                saver3.restore(sess, 'distributed/{}/distributed_{}.ckpt'.format(flag, epoch))



        # logger.warn('Testing evaluation...')
        # saver.restore(sess, tf.train.latest_checkpoint("bigan/train_logs/kdd/{}/{}/{}/central".format(weight, method, random_seed)))
        # inds = rng.permutation(testx.shape[0])
        # testx = testx[inds]  # shuffling  dataset
        # testy = testy[inds] # shuffling  dataset
        # scores = []
        # inference_time = []
        #
        # # Create scores
        # for t in range(nr_batches_test):
        #
        #     # construct randomly permuted minibatches
        #     ran_from = t * batch_size
        #     ran_to = (t + 1) * batch_size
        #     begin_val_batch = time.time()
        #
        #     feed_dict = {input_pl: testx[ran_from:ran_to],
        #                  is_training_pl:False}
        #
        #     scores += sess.run(list_scores,
        #                                  feed_dict=feed_dict).tolist()
        #     inference_time.append(time.time() - begin_val_batch)
        #
        # logger.info('Testing : mean inference time is %.4f' % (
        #     np.mean(inference_time)))
        #
        # ran_from = nr_batches_test * batch_size
        # ran_to = (nr_batches_test + 1) * batch_size
        # size = testx[ran_from:ran_to].shape[0]
        # fill = np.ones([batch_size - size, 121])
        #
        # batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
        # feed_dict = {input_pl: batch,
        #              is_training_pl: False}
        #
        # batch_score = sess.run(list_scores,
        #                    feed_dict=feed_dict).tolist()
        #
        # scores += batch_score[:size]
        #
        # # Highest 80% are anomalous
        # per = np.percentile(scores, 80)
        #
        # y_pred = scores.copy()
        # y_pred = np.array(y_pred)
        #
        # inds = (y_pred < per)
        # inds_comp = (y_pred >= per)
        #
        # y_pred[inds] = 0
        # y_pred[inds_comp] = 1
        #
        #
        # precision, recall, f1,_ = precision_recall_fscore_support(testy,
        #                                                           y_pred,
        #                                                           average='binary')
        #
        # print(
        #     "Testing : Prec = %.4f | Rec = %.4f | F1 = %.4f "
        #     % (precision, recall, f1))

def run(nb_epochs, weight, method, degree, label, random_seed=42, id = 1):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        global client
        if id == 1:
            client = Client(8888)
            client.connectToServer()
        if id == 2:
            client = Client(8889)
            client.connectToServer()
        if id == 3:
            client = Client(8890)
            client.connectToServer()
        tf.set_random_seed(random_seed)
        train_and_test(nb_epochs, weight, method, degree, random_seed, id)