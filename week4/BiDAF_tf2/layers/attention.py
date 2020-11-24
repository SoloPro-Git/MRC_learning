import tensorflow as tf


class C2QAttention(tf.keras.layers.Layer):

    def call(self, similarity, qencode):
        # 1. 对qecncode进行扩展维度 ：tf.expand_dims
        qencode_exp = tf.expand_dims(qencode, axis=1)
        # 2. softmax函数处理相似度矩阵：tf.keras.activations.softmax
        similarity_softmax = tf.keras.activations.softmax(similarity, axis=1)
        # 3. 对处理结果扩展维度：tf.expand_dims
        similarity_softmax_exp = tf.expand_dims(similarity_softmax, axis=-1)
        # 4. 加权求和：tf.math.reduce_sum
        c2q_att = tf.reduce_max(tf.multiply(qencode_exp, similarity_softmax_exp), axis=2)

        return c2q_att


class Q2CAttention(tf.keras.layers.Layer):

    def call(self, similarity, cencode):
        # 1.计算similarity矩阵最大值：tf.math.reduce_max
        simi_max = tf.reduce_max(similarity, axis=2)
        # 2.使用 softmax函数处理最大值的相似度矩阵：tf.keras.activations.softmax
        simi_sfmax = tf.keras.activations.softmax(simi_max, axis=1)
        # 3.维度处理：tf.expand_dims
        simi_sfmax_exp = tf.expand_dims(simi_sfmax, axis=-1)
        # 4.加权求和：tf.math.reduce_sum
        simi_sum = tf.math.reduce_sum(tf.multiply(simi_sfmax_exp, cencode), axis=2)
        # 5.再次维度处理加权求和后的结果：tf.expand_dims
        simi_sum = tf.math.reduce_sum(tf.multiply(simi_sfmax_exp, cencode), axis=2)
        # 6.获取重复的次数： cencode.shape[1]
        # 7.重复拼接获取最终矩阵：tf.tile
        simi_sum_exp = tf.expand_dims(simi_sum, axis=-2)
        q2c_att = tf.tile(simi_sum_exp, (1, cencode.shape[1], 1))
        return q2c_att


if __name__ == '__main__':
    # T=5,J=8 ,2d=10
    g1 = tf.random_uniform_initializer(minval=0)
    simi = g1(shape=[2, 5, 8])
    q = tf.ones(shape=(2, 8, 10))

    att_layer = C2QAttention()
    att_layer.call(simi, q)
