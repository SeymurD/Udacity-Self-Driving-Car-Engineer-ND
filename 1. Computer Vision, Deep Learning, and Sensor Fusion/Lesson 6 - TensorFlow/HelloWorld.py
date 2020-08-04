import tensorflow as tf

# Disable Eager Execution
tf.compat.v1.disable_eager_execution()

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.compat.v1.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
