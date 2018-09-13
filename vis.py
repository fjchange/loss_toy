import tensorflow as tf
import numpy as np
import mnist_loss_toy_openset as mlt
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data

model_path='./models_triplet/model-openset-triplet'
data_file_path='./mnist/mnist_train'
test_data_file_path='./mnist/mnist_test'

saver=tf.train.import_meta_graph(model_path+'-9800.meta')

sess=tf.Session()
saver.restore(sess,tf.train.latest_checkpoint('./models_triplet/'))
graph=sess.graph
input_images=graph.get_tensor_by_name('input:0')
global_step=graph.get_tensor_by_name('global_step:0')

feature=graph.get_tensor_by_name('fc1/BiasAdd:0')

f = plt.figure(figsize=(16, 9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']


images,labels=mlt.get_data(data_file_path,9,0.1)

feat = sess.run(feature, feed_dict={input_images: images})

for i in range(10):
    plt.plot(feat[labels == i, 0].flatten(), feat[labels == i, 1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8','9'])
plt.grid()
plt.show()

images,labels=mlt.get_data(data_file_path,10,0.1)
feat = sess.run(feature, feed_dict={input_images: images})

for i in range(10):
    plt.plot(feat[labels == i, 0].flatten(), feat[labels == i, 1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8','9'])
plt.grid()
plt.show()

images,labels=mlt.get_data(test_data_file_path,10)
feat = sess.run(feature, feed_dict={input_images: images})

for i in range(10):
    plt.plot(feat[labels == i, 0].flatten(), feat[labels == i, 1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8','9'])
plt.grid()
plt.show()

sess.close()