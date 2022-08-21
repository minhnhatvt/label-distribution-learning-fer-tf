import tensorflow as tf
import os

class AdaptiveSimNet(tf.keras.Model):
    def __init__(self, num_neighbors=4, feature_dim=512):
        super(AdaptiveSimNet, self).__init__()
        self.num_neighbors = num_neighbors

        # self.mha = MultiHeadAttention(512, 1)

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(256, kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(1, activation=None),
            tf.keras.layers.Lambda(lambda x: tf.maximum(x, -1.3)),
            tf.keras.layers.Activation('sigmoid')
        ])

    def call(self, feat, feat_aux, training=False):
        '''
        feat: shape (B,D) - D is the hidden dimension
        feat_aux: shape (B,K,D) - K number of neighbors
        return weighting score for neighbors: shape (B,K)
        '''
        # feat_aux = self.mha(feat_aux, training=training)

        B, K, D = feat_aux.shape


        #B, K, D = list(map(int,[B,K,D]))

        feat_repeat = tf.keras.layers.RepeatVector(K)(feat)  # shape (B,K, D)
        x = tf.keras.layers.Concatenate(axis=-1)([feat_repeat, feat_aux])  # shape (B,K,2*D)

        scores = tf.squeeze(self.encoder(x, training=training), axis=-1)
        # scores = tf.sigmoid(tf.reduce_sum(tf.multiply(feat_repeat,feat_aux),axis=-1)/tf.sqrt(512.0))

        return scores
        # return tf.maximum(scores, 0.2)


class ExpNet(tf.keras.Model):
    def __init__(self, num_classes=7, pretrained="msceleb", backbone="resnet50", feature_dim=512):
        super(ExpNet, self).__init__()
        self.num_classes = num_classes

        self.backbone_type = backbone
        if pretrained is None or pretrained == 'imagenet':
            if backbone=="resnet18":
                from classification_models.tfkeras import Classifiers
                ResNet18, preprocess_input = Classifiers.get('resnet18')
                self.backbone = ResNet18(input_shape=(224,224,3), weights=pretrained, include_top=False, pooling="avg")
            elif backbone=="resnet50":
                self.backbone=tf.keras.applications.ResNet50(input_shape=(224,224,3), weights=pretrained, include_top=False, pooling="avg")
            elif backbone=="resnet101":
                self.backbone=tf.keras.applications.resnet.ResNet101(input_shape=(224,224,3), weights=pretrained, include_top=False, pooling="avg")
            elif backbone=="resnet152":
                self.backbone = tf.keras.applications.resnet.Resnet152(input_shape=(224, 224, 3), weights=pretrained,
                                                                       include_top=False, pooling="avg")
        elif pretrained=="msceleb":
            if backbone=="resnet18":
                self.backbone = tf.keras.models.load_model("pretrained/resnet18.h5")
            elif backbone=="resnet50":
                self.backbone = tf.keras.models.load_model("pretrained/resnet50.h5")
            elif backbone=="resnet101":
                self.backbone = tf.keras.models.load_model("pretrained/resnet101.h5")
            elif backbone=="resnet152":
                self.backbone = tf.keras.models.load_model("msceleb_IR_152_Epoch_59.h5")


        else:
            raise ValueError('pretrained type invalid, only supports: None, imagenet, and msceleb')
        self.pretrained=pretrained
        # ================================================================

        # self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax',
                                        kernel_initializer=tf.keras.initializers.HeNormal(),
                                        kernel_regularizer=tf.keras.regularizers.L2(0.001)
                                        )
        self.weighting_net = AdaptiveSimNet(feature_dim)

    def call(self, x, training=False):
        if self.pretrained=="msceleb":
            x = tf.transpose(x, (0, 3, 1, 2))
        feat_map = self.backbone(x, training=training)

        x = feat_map

        out = self.classifier(x)

        return x, out

def create_model(config):
    model = ExpNet(num_classes=config.num_classes, pretrained=config.pretrained , backbone=config.backbone)
    model(tf.ones((32, config.input_size[0], config.input_size[1], 3)))
    model.weighting_net(tf.ones((2, config.feature_dim)), tf.ones((2, 4, config.feature_dim)))
    return model


if __name__=="__main__":
    # model = ExpNet(num_classes=7, pretrained="msceleb", backbone="resnet50")
    # model(tf.ones((32, 112, 112, 3)))
    # model.weighting_net(tf.ones((2, 512)), tf.ones((2, 4, 512)))
    # model.summary()

    model = ExpNet(num_classes=7, pretrained="msceleb", backbone="resnet18")
    print(model(tf.ones((32, 224, 224, 3)))[0].shape)
    model.weighting_net(tf.ones((2, 512)), tf.ones((2, 4, 512)))
    model.summary()







