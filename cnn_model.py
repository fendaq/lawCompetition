import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""
    vocab_dim = 400
    embedding_size = 128  # 词向量维度
    seq_length = 500  # 序列长度
    num_classes = 303  # 类别数
    num_filters = 256  # 卷积核数目
    filter_size = [2, 3, 4, 5]  # 卷积核尺寸
    vocab_size = 4896  # 词汇表大小

    hidden_dim = 1024  # 全连接层神经元

    learning_rate = 1e-4  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 10000  # 总迭代轮次
    kernel_size = 5
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    l2_lambda = 0.1


class CharLevelCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.x = tf.placeholder(
            tf.int32, [None, self.config.seq_length], name='x')
        self.y = tf.placeholder(
            tf.float32, [None, self.config.num_classes], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.l2_loss = 0
        self.l2_lambda = 0.2

        self.char_level_cnn()

    def char_level_cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable(
                'embedding', [self.config.vocab_size, self.config.embedding_size])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.x)

        with tf.name_scope("cnn_1"):
            # CNN layer
            conv_1 = tf.layers.conv1d(
                embedding_inputs, 64, 3, name='conv1', trainable=True, padding='same')
            conv_1 = tf.nn.relu(conv_1)
            max_pool_1 = tf.layers.max_pooling1d(
                conv_1, 2, strides=1, padding='same')
            conv_2 = tf.layers.conv1d(max_pool_1, 64, 1, name='conv2')
            conv_3 = tf.layers.conv1d(conv_2, 32, 3, name='conv3')
            max_pool_3 = tf.layers.max_pooling1d(
                conv_3, 1, strides=1, padding='same')
            gmp = tf.reduce_max(max_pool_3, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            W = tf.get_variable("W", shape=[
                                self.config.hidden_dim, self.config.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(fc, W, b)
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.y)
            self.loss = tf.reduce_mean(
                cross_entropy)+self.l2_loss*self.l2_lambda
            # 优化器
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
<<<<<<< HEAD
            correct_pred = tf.equal(self.logits, self.y)
=======
            correct_pred = tf.equal(self.y_pred,tf.argmax(self.y,1))
>>>>>>> 879e41daef59f27cdc35451afa4598924a7f412c
            self.precision = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
