import tensorflow as tf
a = tf.constant(1.3, name='const_A')
b = tf.Variable(3.1, name='b')
c = tf.add(a, b, name='addition')
d = tf.multiply(c, a, name='multiply')

for op in tf.get_default_graph().get_operations():
    print str(op.name) 