# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/3/4
# version： Python 3.7.8
# @File : test.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import math


target = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
predict = np.array([0.1, 0.6, 0.3, 0, 0, 0, 0, 0, 0, 0])
predict = tf.convert_to_tensor(predict)
from keras import backend as K
loss= K.categorical_crossentropy(target=target, output=predict, from_logits=False)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(y)*y',axis=1)))
#loss = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=target)

with tf.Session() as sess:
    print(sess.run(loss))
    print(math.log(0.6))