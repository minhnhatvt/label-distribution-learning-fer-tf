import tensorflow as tf
import numpy as np

def create_sample_weights(y_batch, class_weights_dict = None):
    if class_weights_dict == None:
        return tf.ones(tf.shape(y_batch)[0], dtype=tf.float32)
    keys_tensor = tf.convert_to_tensor(list(class_weights_dict.keys()), dtype=tf.int32)
    vals_tensor = tf.convert_to_tensor(list(class_weights_dict.values()), dtype=tf.float32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=-1)
    return table.lookup(tf.cast(y_batch,dtype=tf.int32))

def construct_target_distribution(l_gt, y_aux, knn_weights, calib_scores, lamb = 0.9, NUM_CLASSES = 7):
    '''
    Input:
        l_gt: groundtruth label - shape (B,)
        y_aux:  predicted neighbor distribution - shape (B,K,C)
        knn_weights - shape (B,K)
        calib_scores: predicted calibration score for each neighbors  - shape (B,K)
        lamb: the balanced coefficients - scalar or vector with shape (B,1)
    Return target label distribution - shape (B,C)
    '''

    B,K = knn_weights.shape
    knn_weights = tf.multiply(knn_weights,calib_scores)
    neighbor_ld = tf.squeeze(tf.linalg.matmul(y_aux, tf.cast(tf.expand_dims(knn_weights,-1),dtype=tf.float32), transpose_a=True),axis=-1) #shape (B,C)
    neighbor_ld = neighbor_ld / tf.reduce_sum(knn_weights,axis=-1,keepdims=True)
    y = lamb * tf.one_hot(l_gt, depth=NUM_CLASSES) + (1-lamb) * neighbor_ld

    return y


class DiscriminativeLoss(tf.keras.layers.Layer):
    def __init__(self, num_classes, feature_dim, centers_alpha=0.5, init_centers=None):
        super(DiscriminativeLoss, self).__init__()
        self.centers = init_centers
        if init_centers is None:
            self.centers = tf.Variable(tf.transpose(tf.keras.initializers.constant(0)((feature_dim, num_classes))),
                                       name='centers')  # shape (C,D)
        self.feat_dim = feature_dim
        self.num_classes = num_classes
        self.optimizer = tf.keras.optimizers.SGD(lr=centers_alpha)

    def call(self, x, y, x_aux, y_aux, lamb_hat, indices):
        '''
            x: input feature vector: shape (B,D)
            x_aux: neighbor feature vector shape (B,K,D)
            y_aux: int label array with shape (B,K)
        '''

        B, D = x.shape
        x_aux = tf.reshape(x_aux, (-1, D))  # shape (B*K,D)
        self.x = tf.concat([x, x_aux], axis=0)
        y_aux = tf.reshape(y_aux, shape=-1)  # shape (B*K)
        self.y = tf.concat([y, y_aux], axis=0)

        lamb = tf.gather(lamb_hat, indices)
        lamb = tf.sigmoid(lamb)  # shape (B,1)
        self.lamb = tf.stop_gradient(lamb)

        # self.lamb=1

        B, D = tf.shape(self.x)
        centers_batch = tf.gather(self.centers, self.y)  # shape (B,D)
        dist = tf.square(self.x - centers_batch)  # shape (B,D)

        center_loss = (0.5 * tf.reduce_sum(self.lamb * dist)) / tf.cast(B, tf.float32)


        return center_loss

    def compute_grads(self):
        B, D = tf.shape(self.x)
        centers_batch = tf.gather(self.centers, self.y)  # shape (B,D)
        diff = self.x - centers_batch  # shape (B,D)
        y_onehot = tf.one_hot(self.y, depth=self.num_classes)
        return tf.matmul(tf.transpose(y_onehot), -self.lamb * diff) / (
                    tf.expand_dims(tf.reduce_sum(tf.transpose(y_onehot), axis=1), axis=-1) + 1) / tf.cast(B, tf.float32)

    def update_centers(self, grads):
        self.optimizer.apply_gradients([(grads, self.centers)])

        # alternative update l2 distance between centers and mu
        grads = 0

        SB = 0
        with tf.GradientTape() as t:
            t.watch(self.centers)
            mu = tf.reduce_mean(self.centers, axis=0, keepdims=False)
            # l2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.centers-mu),axis=1))
            for i in range(self.num_classes):
                for j in range(i, self.num_classes):
                    if i != j:
                        dist = tf.reduce_sum(tf.square(self.centers[i] - self.centers[j]))  # scalar
                        SB = SB + tf.exp(-dist / self.feat_dim)
            grads = t.gradient(SB, self.centers)  # shape (C,D)

        grads = 0.1 * grads

        self.optimizer.apply_gradients([(grads, self.centers)])
        # ============================


def CELoss(y_true, y_pred, sample_weight=None):
    #reimplement the cross entropy loss since the existing one in tf has an extra normalization step which can lead to incorrect gradients
    if sample_weight is not None:
        return tf.reduce_mean(tf.reduce_sum(-y_true * tf.math.log(y_pred + 1e-10), axis=-1) * sample_weight)
    return tf.reduce_mean(tf.reduce_sum(-y_true * tf.math.log(y_pred + 1e-10), axis=-1))

def get_optimizer(train_dataset, config):
    batches_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    lr_init = config.lr
    lr_decay = config.lr_decay
    decay_steps = np.array(config.lr_steps) * batches_per_epoch
    lrs = np.arange(decay_steps.shape[0] + 1)
    lrs = lr_init * (lr_decay ** lrs)

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        list(decay_steps), list(lrs))
    if config.optimizer=='adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif config.optimizer=='sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    else:
        raise ValueError("Optimizer not supported!")
    return optimizer


