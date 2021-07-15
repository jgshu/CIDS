import time
import numpy as np
import tensorflow as tf
import logging
import importlib
import sys
import bigan.kdd_utilities as network
import data.kdd_half as data
import data.kdd as data1
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import time

RANDOM_SEED = 13
FREQ_PRINT = 20 # print frequency image tensorboard [20]


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
    return "bigan/train_logs/kdd/{}/{}/{}".format(weight, method, rd)

def create_central_logdir(method, weight, rd):
    """ Directory to save training logs, weights, biases, etc."""
    return "bigan/train_logs/kdd/{}/{}/{}/central/central".format(weight, method, rd)

def create_distributed_logdir(method, weight, rd, id):
    """ Directory to save training logs, weights, biases, etc."""
    return "bigan/train_logs/kdd/{}/{}/{}/distributed/{}".format(weight, method, rd, id, id)

def test(nb_epochs, weight, method, degree, random_seed):
    logger = logging.getLogger("BiGAN.train.kdd.{}".format(method))
    # Placeholders
    input_pl = tf.placeholder(tf.float32, shape=data.get_shape_input(), name="input")
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate = tf.placeholder(tf.float32, shape=(), name="lr_pl")

    # Data
    # trainx, trainy = data1.get_train()
    # trainx_copy = trainx.copy()
    testx, testy = data1.get_test()

    # Parameters
    starting_lr = network.learning_rate
    batch_size = network.batch_size
    latent_dim = network.latent_dim
    ema_decay = 0.9999

    rng = np.random.RandomState(RANDOM_SEED)
    # nr_batches_train = int(trainx.shape[0] / batch_size)
    nr_batches_test = int(testx.shape[0] / batch_size)

    logger.info('Building training graph...')

    logger.warn("The BiGAN is training with the following parameters:")
    display_parameters(batch_size, starting_lr, ema_decay, weight, method, degree)

    gen = network.decoder
    enc = network.encoder
    dis = network.discriminator

    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)

    with tf.variable_scope('generator_model'):
        z1 = tf.random_normal([batch_size, latent_dim])
        # z = tf.random_normal([batch_size, latent_dim])
        z = tf.placeholder(tf.float32, shape=(batch_size, latent_dim), name="z")
        x_gen = gen(z, is_training=is_training_pl)



    with tf.variable_scope('discriminator_model'):
        l_encoder, inter_layer_inp = dis(z_gen, input_pl, is_training=is_training_pl)
        z_ = tf.placeholder(tf.float32, shape=(50, 32), name="z_")

        gen_ = tf.placeholder(tf.float32, shape=(50, 121), name="gen_")
        l_generator, inter_layer_rct = dis(z_, gen_, is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        # discriminator
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder),logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),logits=l_generator))
        loss_discriminator = loss_dis_gen + loss_dis_enc
        # generator
        l_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator), logits=l_generator))
        cross_g = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator), logits=l_generator)
        x = tf.gradients(cross_g, gen_)
        g1 = tf.placeholder(tf.float32, shape=(50, 121), name="g1")
        g2 = tf.placeholder(tf.float32, shape=(50, 121), name="g2")
        total = 0
        for i in range(50):
            v_g = tf.slice(g1, [i, 0], [1, 121])
            v_x_gen = tf.slice(x_gen, [i, 0], [1, 121])
            v_g = tf.reshape(v_g, [121, 1])
            temp = tf.matmul(v_x_gen, v_g)
            total += temp
        for i in range(50):
            v_g = tf.slice(g2, [i, 0], [1, 121])
            v_x_gen = tf.slice(x_gen, [i, 0], [1, 121])
            v_g = tf.reshape(v_g, [121, 1])
            temp = tf.matmul(v_x_gen, v_g)
            total += temp
        loss_generator = total / 100
        # encoder
        loss_encoder = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder), logits=l_encoder))


    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for estimation
        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)

        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')

    logger.info('Building testing graph...')

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(input_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True)

    with tf.variable_scope('generator_model'):
        reconstruct_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_model'):
        l_encoder_ema, inter_layer_inp_ema = dis(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema),
                                                 reuse=True)
        l_generator_ema, inter_layer_rct_ema = dis(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema),
                                                   reuse=True)
    with tf.name_scope('Testing'):
        with tf.variable_scope('Reconstruction_loss'):
            delta = input_pl - reconstruct_ema
            delta_flat = tf.contrib.layers.flatten(delta)
            gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                              keep_dims=False, name='epsilon')

        with tf.variable_scope('Discriminator_loss'):
            if method == "cross-e":
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator_ema),logits=l_generator_ema)

            elif method == "fm":
                fm = inter_layer_inp_ema - inter_layer_rct_ema
                fm = tf.contrib.layers.flatten(fm)
                dis_score = tf.norm(fm, ord=degree, axis=1,
                                 keep_dims=False, name='d_loss')

            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Score'):
            list_scores = (1 - weight) * gen_score + weight * dis_score

    variables = tf.contrib.framework.get_variables_to_restore()
    # restore_variable_list1=['encoder_model/encoder/layer_1/fc/kernel:0',
    #                         'encoder_model/encoder/layer_1/fc/bias:0',
    #                         'encoder_model/encoder/layer_2/fc/kernel:0',
    #                         'encoder_model/encoder/layer_2/fc/bias:0',
    #                         'encoder_model/encoder/layer_1/fc/kernel/enc_optimizer:0',
    #                         'encoder_model/encoder/layer_1/fc/kernel/enc_optimizer_1:0',
    #                         'encoder_model/encoder/layer_1/fc/bias/enc_optimizer:0',
    #                         'encoder_model/encoder/layer_1/fc/bias/enc_optimizer_1:0',
    #                         'encoder_model/encoder/layer_2/fc/kernel/enc_optimizer:0',
    #                         'encoder_model/encoder/layer_2/fc/kernel/enc_optimizer_1:0',
    #                         'encoder_model/encoder/layer_2/fc/bias/enc_optimizer:0',
    #                         'encoder_model/encoder/layer_2/fc/bias/enc_optimizer_1:0',
    #                         'encoder_model/encoder/layer_1/fc/kernel/ExponentialMovingAverage:0',
    #                         'encoder_model/encoder/layer_1/fc/bias/ExponentialMovingAverage:0',
    #                         'encoder_model/encoder/layer_2/fc/kernel/ExponentialMovingAverage:0',
    #                         'encoder_model/encoder/layer_2/fc/bias/ExponentialMovingAverage:0']
    # restore_variable_list2=['generator_model/generator/layer_1/fc/kernel:0',
    #                         'generator_model/generator/layer_1/fc/bias:0',
    #                         'generator_model/generator/layer_2/fc/kernel:0',
    #                         'generator_model/generator/layer_2/fc/bias:0',
    #                         'generator_model/generator/layer_3/fc/kernel:0',
    #                         'generator_model/generator/layer_3/fc/bias:0',
    #                         'generator_model/generator/layer_1/fc/kernel/gen_optimizer:0',
    #                         'generator_model/generator/layer_1/fc/kernel/gen_optimizer_1:0',
    #                         'generator_model/generator/layer_1/fc/bias/gen_optimizer:0',
    #                         'generator_model/generator/layer_1/fc/bias/gen_optimizer_1:0',
    #                         'generator_model/generator/layer_2/fc/kernel/gen_optimizer:0',
    #                         'generator_model/generator/layer_2/fc/kernel/gen_optimizer_1:0',
    #                         'generator_model/generator/layer_2/fc/bias/gen_optimizer:0',
    #                         'generator_model/generator/layer_2/fc/bias/gen_optimizer_1:0',
    #                         'generator_model/generator/layer_3/fc/kernel/gen_optimizer:0',
    #                         'generator_model/generator/layer_3/fc/kernel/gen_optimizer_1:0',
    #                         'generator_model/generator/layer_3/fc/bias/gen_optimizer:0',
    #                         'generator_model/generator/layer_3/fc/bias/gen_optimizer_1:0',
    #                         'generator_model/generator/layer_1/fc/kernel/ExponentialMovingAverage:0',
    #                         'generator_model/generator/layer_1/fc/bias/ExponentialMovingAverage:0',
    #                         'generator_model/generator/layer_2/fc/kernel/ExponentialMovingAverage:0',
    #                         'generator_model/generator/layer_2/fc/bias/ExponentialMovingAverage:0',
    #                         'generator_model/generator/layer_3/fc/kernel/ExponentialMovingAverage:0',
    #                         'generator_model/generator/layer_3/fc/bias/ExponentialMovingAverage:0']
    # restore_variable_list3=['discriminator_model/discriminator/x_layer_1/fc/kernel:0',
    #                         'discriminator_model/discriminator/x_layer_1/fc/bias:0',
    #                         'discriminator_model/discriminator/z_fc_1/dense/kernel:0',
    #                         'discriminator_model/discriminator/z_fc_1/dense/bias:0',
    #                         'discriminator_model/discriminator/y_fc_1/dense/kernel:0',
    #                         'discriminator_model/discriminator/y_fc_1/dense/bias:0',
    #                         'discriminator_model/discriminator/y_fc_logits/dense/kernel:0',
    #                         'discriminator_model/discriminator/y_fc_logits/dense/bias:0',
    #                         'discriminator_model/discriminator/x_layer_1/fc/kernel/dis_optimizer:0',
    #                         'discriminator_model/discriminator/x_layer_1/fc/kernel/dis_optimizer_1:0',
    #                         'discriminator_model/discriminator/x_layer_1/fc/bias/dis_optimizer:0',
    #                         'discriminator_model/discriminator/x_layer_1/fc/bias/dis_optimizer_1:0',
    #                         'discriminator_model/discriminator/z_fc_1/dense/kernel/dis_optimizer:0',
    #                         'discriminator_model/discriminator/z_fc_1/dense/kernel/dis_optimizer_1:0',
    #                         'discriminator_model/discriminator/z_fc_1/dense/bias/dis_optimizer:0',
    #                         'discriminator_model/discriminator/z_fc_1/dense/bias/dis_optimizer_1:0',
    #                         'discriminator_model/discriminator/y_fc_1/dense/kernel/dis_optimizer:0',
    #                         'discriminator_model/discriminator/y_fc_1/dense/kernel/dis_optimizer_1:0',
    #                         'discriminator_model/discriminator/y_fc_1/dense/bias/dis_optimizer:0',
    #                         'discriminator_model/discriminator/y_fc_1/dense/bias/dis_optimizer_1:0',
    #                         'discriminator_model/discriminator/y_fc_logits/dense/kernel/dis_optimizer:0',
    #                         'discriminator_model/discriminator/y_fc_logits/dense/kernel/dis_optimizer_1:0',
    #                         'discriminator_model/discriminator/y_fc_logits/dense/bias/dis_optimizer:0',
    #                         'discriminator_model/discriminator/y_fc_logits/dense/bias/dis_optimizer_1:0',
    #                         'discriminator_model/discriminator/x_layer_1/fc/kernel/ExponentialMovingAverage:0',
    #                         'discriminator_model/discriminator/x_layer_1/fc/bias/ExponentialMovingAverage:0',
    #                         'discriminator_model/discriminator/z_fc_1/dense/kernel/ExponentialMovingAverage:0',
    #                         'discriminator_model/discriminator/z_fc_1/dense/bias/ExponentialMovingAverage:0',
    #                         'discriminator_model/discriminator/y_fc_1/dense/kernel/ExponentialMovingAverage:0',
    #                         'discriminator_model/discriminator/y_fc_1/dense/bias/ExponentialMovingAverage:0',
    #                         'discriminator_model/discriminator/y_fc_logits/dense/kernel/ExponentialMovingAverage:0',
    #                         'discriminator_model/discriminator/y_fc_logits/dense/bias/ExponentialMovingAverage:0']
    central_dir = create_central_logdir(weight, method, random_seed)+'central.ckpt'
    id_1 = 1
    distrbuted_dir = create_distributed_logdir(weight, method, random_seed, id_1) + 'distributed_{}.ckpt'.format(id_1)

    restore_variable_list1 = [v for v in variables if 'generator_model' in v.name]
    restore_variable_list2 = [v for v in variables if 'encoder_model' in v.name]
    restore_variable_list3 = [v for v in variables if 'discriminator_model' in v.name]
    # for i in restore_variable_list1:
    #     print(i)
    saver1 = tf.train.Saver(restore_variable_list1)
    saver2 = tf.train.Saver(restore_variable_list2)
    saver3 = tf.train.Saver(restore_variable_list3)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        logger.warn('Testing evaluation...')
        # print(central_dir)
        # print(distrbuted_dir)
        for i in range(50):
            saver1.restore(sess, 'central/central_{}.ckpt'.format(i+1))
            saver2.restore(sess, 'distributed/2/distributed_{}.ckpt'.format(i+1))
            saver3.restore(sess, 'distributed/2/distributed_{}.ckpt'.format(i+1))
            # saver1.restore(sess, 'model/model_1_{}.ckpt'.format((i+1)*5))
            # saver2.restore(sess, 'model/model_1_{}.ckpt'.format((i+1)*5))
            # saver3.restore(sess, 'model/model_1_{}.ckpt'.format((i+1)*5))
            # saver.restore(sess, 'model/model_1_{}.ckpt'.format((i+1)*5))
            # init = tf.global_variables_initializer()  # 初始化所在的位置至关重要，以本程序为例，使用adam优化器时，会主动创建变量。
            # # 因此，如果这时的初始化位置在创建adam优化器之前，则adam中包含的变量会未初始化，然后报错。本行初始化时，可以看到Adam
            # # 已经声明，古不会出错
            # sess.run(init)
            inds = rng.permutation(testx.shape[0])
            testx = testx[inds]  # shuffling  dataset
            testy = testy[inds]  # shuffling  dataset
            scores = []
            inference_time = []

            # Create scores
            for t in range(nr_batches_test):
                # construct randomly permuted minibatches
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size
                begin_val_batch = time.time()

                feed_dict = {input_pl: testx[ran_from:ran_to],
                             is_training_pl: False}

                scores += sess.run(list_scores,
                                   feed_dict=feed_dict).tolist()
                inference_time.append(time.time() - begin_val_batch)

            ran_from = nr_batches_test * batch_size
            ran_to = (nr_batches_test + 1) * batch_size
            size = testx[ran_from:ran_to].shape[0]
            fill = np.ones([batch_size - size, 121])

            batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
            feed_dict = {input_pl: batch,
                         is_training_pl: False}

            batch_score = sess.run(list_scores,
                                   feed_dict=feed_dict).tolist()

            scores += batch_score[:size]

            # Highest 80% are anomalous
            per = np.percentile(scores, 80)

            y_pred = scores.copy()
            y_pred = np.array(y_pred)

            inds = (y_pred < per)
            inds_comp = (y_pred >= per)

            y_pred[inds] = 0
            y_pred[inds_comp] = 1

            accuracy = metrics.accuracy_score(testy, y_pred)
            precision = metrics.precision_score(testy, y_pred)
            recall = metrics.recall_score(testy, y_pred)
            f1 = metrics.f1_score(testy, y_pred)

            print(
                "epoch%d:Testing : Acc = %.4f | Prec = %.4f | Rec = %.4f | F1 = %.4f "
                % ((i+1), accuracy, precision, recall, f1))



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
    # trainx1, trainy1 = data1.get_train(1)
    # # trainx_copy1 = trainx1.copy()
    # # testx1, testy1 = data.get_test(1)
    #
    # trainx2, trainy2 = data.get_train(2)
    #
    # # trainx_copy1 = trainx1.copy()
    # # testx2, testy2 = data.get_test(2)
    # trainx = np.concatenate((trainx1, trainx2), axis=0)
    # trainy = np.concatenate((trainy1, trainy2), axis=0)
    # trainx_copy = trainx.copy()

    trainx, trainy = data.get_train(3)
    print(trainx.shape)
    trainx_copy = trainx.copy()
    testx, testy = data1.get_test()
    print(testx.shape)


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
    enc = network.encoder
    dis = network.discriminator

    with tf.variable_scope('encoder_model'):
        z_gen = enc(input_pl, is_training=is_training_pl)

    with tf.variable_scope('generator_model'):
        z1 = tf.random_normal([batch_size, latent_dim])
        # z = tf.random_normal([batch_size, latent_dim])
        z = tf.placeholder(tf.float32, shape=(batch_size, latent_dim), name="z")
        x_gen = gen(z, is_training=is_training_pl)



    with tf.variable_scope('discriminator_model'):
        l_encoder, inter_layer_inp = dis(z_gen, input_pl, is_training=is_training_pl)
        z_ = tf.placeholder(tf.float32, shape=(50, 32), name="z_")
        gen_ = tf.placeholder(tf.float32, shape=(50, 121), name="gen_")
        l_generator, inter_layer_rct = dis(z_, gen_, is_training=is_training_pl, reuse=True)

    with tf.name_scope('loss_functions'):
        # discriminator
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder),logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),logits=l_generator))
        loss_discriminator = loss_dis_gen + loss_dis_enc
        # generator
        l_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator), logits=l_generator))
        cross_g = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator), logits=l_generator)
        x = tf.gradients(cross_g, gen_)
        g = tf.placeholder(tf.float32, shape=(50, 121), name="g")
        total = 0.0
        for i in range(50):
            v_g = tf.slice(g, [i, 0], [1, 121])
            v_x_gen = tf.slice(x_gen, [i, 0], [1, 121])
            v_g = tf.reshape(v_g, [121, 1])
            temp = tf.matmul(v_x_gen, v_g)
            total += temp
        print("total:", total)
        loss_generator = total/50
        # encoder
        loss_encoder = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder), logits=l_encoder))


    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='enc_optimizer')

        with tf.control_dependencies(update_ops_gen):
            gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

        # Exponential Moving Average for estimation
        dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_dis = dis_ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op_dis)

        gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_gen = gen_ema.apply(gvars)

        with tf.control_dependencies([gen_op]):
            train_gen_op = tf.group(maintain_averages_op_gen)

        enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        maintain_averages_op_enc = enc_ema.apply(evars)

        with tf.control_dependencies([enc_op]):
            train_enc_op = tf.group(maintain_averages_op_enc)

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')

    logger.info('Building testing graph...')

    with tf.variable_scope('encoder_model'):
        z_gen_ema = enc(input_pl, is_training=is_training_pl,
                        getter=get_getter(enc_ema), reuse=True)

    with tf.variable_scope('generator_model'):
        reconstruct_ema = gen(z_gen_ema, is_training=is_training_pl,
                              getter=get_getter(gen_ema), reuse=True)

    with tf.variable_scope('discriminator_model'):
        l_encoder_ema, inter_layer_inp_ema = dis(z_gen_ema,
                                                 input_pl,
                                                 is_training=is_training_pl,
                                                 getter=get_getter(dis_ema),
                                                 reuse=True)
        l_generator_ema, inter_layer_rct_ema = dis(z_gen_ema,
                                                   reconstruct_ema,
                                                   is_training=is_training_pl,
                                                   getter=get_getter(dis_ema),
                                                   reuse=True)
    with tf.name_scope('Testing'):
        with tf.variable_scope('Reconstruction_loss'):
            delta = input_pl - reconstruct_ema
            delta_flat = tf.contrib.layers.flatten(delta)
            gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                              keep_dims=False, name='epsilon')

        with tf.variable_scope('Discriminator_loss'):
            if method == "cross-e":
                dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(l_generator_ema),logits=l_generator_ema)

            elif method == "fm":
                fm = inter_layer_inp_ema - inter_layer_rct_ema
                fm = tf.contrib.layers.flatten(fm)
                dis_score = tf.norm(fm, ord=degree, axis=1,
                                 keep_dims=False, name='d_loss')

            dis_score = tf.squeeze(dis_score)

        with tf.variable_scope('Score'):
            list_scores = (1 - weight) * gen_score + weight * dis_score


    logdir = create_logdir(weight, method, random_seed)
    print(tf.contrib.framework.get_variables_to_restore())
    # sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=None,
    #                          save_model_secs=120)
    logger.info('Start training...')
    saver = tf.train.Saver(max_to_keep=1000, keep_checkpoint_every_n_hours=1)
    with tf.Session() as sess:

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
            time_data, time_generator, time_discriminate, time_encoder = [0, 0, 0, 0]
            # training
            for t in range(nr_batches_train):
                
                display_progression_epoch(t, nr_batches_train)             
                ran_from = t * batch_size
                ran_to = (t + 1) * batch_size

                # start = time.clock()
                z1_ = sess.run(z1)
                z1_ = np.array(z1_)
                feed_dict = {is_training_pl: True,
                             z: z1_}
                x_gen_ = sess.run([x_gen], feed_dict=feed_dict)
                x_gen_ = np.array(x_gen_)
                x_gen_ = x_gen_.reshape(50, 121)
                # end = time.clock()
                # print("time_data:", end-start)
                # time_data += end-start

                # start = time.clock()
                # train discriminator
                feed_dict = {input_pl: trainx[ran_from:ran_to],
                             is_training_pl: True,
                             z_: z1_,
                             gen_: x_gen_,
                             learning_rate: lr}

                _, ld, x_, lg, sm = sess.run([train_dis_op,
                                      loss_discriminator,
                                      x,
                                      l_g,
                                      sum_op_dis],
                                     feed_dict=feed_dict)

                train_loss_dis += ld
                writer.add_summary(sm, train_batch)
                # end = time.clock()
                # print("time_discriminate:", end-start)
                # time_discriminate += end - start

                # start = time.clock()
                # train generator
                x_ = np.array(x_)
                x_ = x_.reshape(50, 121)
                feed_dict = {g: x_,
                             z: z1_,
                             is_training_pl: True,
                             learning_rate: lr}
                _ = sess.run(train_gen_op, feed_dict=feed_dict)
                # end = time.clock()
                # print("time_generator:", end-start)
                # time_generator += end-start

                # start = time.clock()
                # train encoder
                feed_dict = {input_pl: trainx_copy[ran_from:ran_to],
                             is_training_pl: True,
                             learning_rate: lr}
                _, le = sess.run([train_enc_op,
                                    loss_encoder
                                    ],
                                    feed_dict=feed_dict)
                # end = time.clock()
                # print("time_encoder:", end-start)
                # time_encoder += end-start

                train_loss_gen += lg
                train_loss_enc += le
                # writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis /= nr_batches_train

            # logger.info('Epoch terminated')
            # print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f "
            #       % (epoch, time.time() - begin, train_loss_gen, train_loss_enc, train_loss_dis))
            # print("time_data = %.4f | time_generator = %.4f | time_discriminate = %.4f | time_encoder = %.4f"
            #       % (time_data, time_generator, time_discriminate, time_encoder))

            epoch += 1

            if epoch % 5 == 0:
                # print("save model parameter")
                save_path = saver.save(sess, 'model/model_1_{}.ckpt'.format(epoch))
                # print('模型保存到:', save_path)
                # logger.warn('Testing evaluation...')

                inds = rng.permutation(testx.shape[0])
                testx = testx[inds]  # shuffling  dataset
                testy = testy[inds] # shuffling  dataset
                scores = []
                inference_time = []

                # Create scores
                for t in range(nr_batches_test):

                    # construct randomly permuted minibatches
                    ran_from = t * batch_size
                    ran_to = (t + 1) * batch_size
                    begin_val_batch = time.time()

                    feed_dict = {input_pl: testx[ran_from:ran_to],
                                 is_training_pl:False}

                    scores += sess.run(list_scores,
                                                 feed_dict=feed_dict).tolist()
                    inference_time.append(time.time() - begin_val_batch)

                # logger.info('Testing : mean inference time is %.4f' % (
                #     np.mean(inference_time)))

                ran_from = nr_batches_test * batch_size
                ran_to = (nr_batches_test + 1) * batch_size
                size = testx[ran_from:ran_to].shape[0]
                fill = np.ones([batch_size - size, 121])

                batch = np.concatenate([testx[ran_from:ran_to], fill], axis=0)
                feed_dict = {input_pl: batch,
                             is_training_pl: False}

                batch_score = sess.run(list_scores,
                                   feed_dict=feed_dict).tolist()

                scores += batch_score[:size]

                # Highest 80% are anomalous
                per = np.percentile(scores, 80)

                y_pred = scores.copy()
                y_pred = np.array(y_pred)

                inds = (y_pred < per)
                inds_comp = (y_pred >= per)

                y_pred[inds] = 0
                y_pred[inds_comp] = 1

                accuracy = metrics.accuracy_score(testy, y_pred)
                precision = metrics.precision_score(testy, y_pred)
                recall = metrics.recall_score(testy, y_pred)
                f1 = metrics.f1_score(testy, y_pred)

                # precision, recall, f1,_ = precision_recall_fscore_support(testy,
                #                                                           y_pred,
                #                                                           average='binary')

                print(
                    "epoch %d:Testing : Acc = %.4f | Prec = %.4f | Rec = %.4f | F1 = %.4f "
                    % (epoch, accuracy, precision, recall, f1))



def run(nb_epochs, weight, method, degree, label, random_seed=42):
    """ Runs the training process"""
    with tf.Graph().as_default():
        # Set the graph level seed
        tf.set_random_seed(random_seed)
        # train_and_test(nb_epochs, weight, method, degree, random_seed)
        test(nb_epochs, weight, method, degree, random_seed)