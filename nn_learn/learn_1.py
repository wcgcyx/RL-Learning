import tensorflow as tf
import numpy as np

# Create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 2 + 0.3

# Create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 1D data, range from -1 to 1
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)  # Learning rate
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
# Create tensorflow structure end

sess = tf.Session()
sess.run(init)      # Very important

for step in range(400):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))