#coding=utf-8
import tensorflow as tf
import os
import tflearn
import random
import numpy as np
import PIL.Image as Image
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import data_flow_ops

#softmax loss 与 center loss的比重调节
LAMBDA=0.5
#center的更新权重
CENTER_LOSS_ALPHA=0.5
NUM_CLASSES=9
model_dir='./models/model-openset'
test_data_file_path='./mnist/mnist_test'
data_file_path='./mnist/mnist_train'
BATCH_SIZE=128
slim=tf.contrib.slim
#--------gpu use only----------
#os.environ['CUDA_VISIBLE_DEVICES']='0'

global_step=tf.Variable(0,trainable=False,name='global_step')

def get_center_loss(features,labels,alpha,num_class):
    len_features=features.shape[1]
    #获取上一次训练得出的center的位置
    centers=tf.get_variable('centers',[num_class,len_features],dtype=tf.float32,
                            initializer=tf.constant_initializer(0),trainable=False)
    labels=tf.reshape(labels,[-1])
    #筛选出相关的类的中心
    centers_batch=tf.gather(centers,labels)

    loss=tf.nn.l2_loss(features-centers_batch)
    #center更新程度
    diff=centers_batch-features
    unique_label,unique_idx,unique_count=tf.unique_with_counts(labels)
    appear_times=tf.gather(unique_count,unique_idx)
    appear_times=tf.reshape(appear_times,[-1,1])

    diff=diff/tf.cast((1+appear_times),tf.float32)
    diff=alpha*diff

    centers_update_op=tf.scatter_sub(centers,labels,diff)

    return loss,centers,centers_update_op

def inference(input_images):
    x=tf.layers.conv2d(inputs=input_images,filters=32,kernel_size=3,padding='SAME',name='conv1_1')
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, padding='SAME',name='conv1_2')
    x=tf.layers.max_pooling2d(inputs=x,pool_size=2,strides=2,name='pool1')

    x=tf.layers.conv2d(inputs=x,filters=64,kernel_size=3,padding='SAME',name='conv2_1')
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=3, padding='SAME',name='conv2_2')
    x=tf.layers.max_pooling2d(inputs=x,pool_size=2,strides=2,name='pool2')

    x=tf.layers.conv2d(inputs=x,filters=128,kernel_size=3,padding='SAME',name='conv3_1')
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=3, padding='SAME',name='conv3_2')
    x=tf.layers.max_pooling2d(inputs=x,pool_size=2,strides=2,name='pool3')

    x=tf.layers.flatten(inputs=x,name='flatten')

    feature=tf.layers.dense(inputs=x,units=2,name='fc1')
    x=tflearn.prelu(feature)

    x=tf.layers.dense(inputs=x,units=NUM_CLASSES,name='fc2')
    return x,feature

def build_network(input_images,labels,ratio=0.5):
    logits,features=inference(input_images)

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss,centers,centers_update_op=get_center_loss(features,labels,CENTER_LOSS_ALPHA,NUM_CLASSES)
        with tf.name_scope('softmax_loss'):
            softmax_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
        with tf.name_scope('total_loss'):
            total_loss=softmax_loss+ratio*center_loss

    with tf.name_scope('acc'):
        accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits,1),labels),tf.float32))

    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss',center_loss)
        tf.summary.scalar('SoftmaxLoss',softmax_loss)
        tf.summary.scalar('TotalLoss',total_loss)

    return logits,features,total_loss,accuracy,centers_update_op

def get_data(file_path,RANGE=NUM_CLASSES,ratio=1.0):
    images=[]
    labels=[]
    for i in range(RANGE):
        tmp=os.path.join(file_path,str(i))
        for _root,_dirs,_paths in os.walk(tmp):
            for _path in _paths:
                if random.random()>=ratio:
                    continue
                temp=os.path.join(tmp,_path)
                img = Image.open(temp)
                img_np = np.array(img)
                img_np=img_np.reshape((28,28,1))
                images.append(img_np)
                labels.append(i)
    images=np.array(images)
    mean_data=np.mean(images,axis=0)
    images=(images-mean_data)/255
    labels=np.array(labels)
    return images,labels

def get_batch(start_index,indexes,images,labels):
    image_batch=[]
    label_batch=[]
    for i in range(BATCH_SIZE):
        image_batch.append(images[indexes[(start_index+i)%len(indexes)]])
        label_batch.append(labels[indexes[(start_index+i)%len(indexes)]])
    return (start_index+BATCH_SIZE)%len(indexes),image_batch,label_batch

#----------------------main-----------------------
if __name__=='__main__':
    images,labels=get_data(data_file_path)
    test_images,test_labels=get_data(test_data_file_path)

    indexes=np.arange(len(images))
    random.shuffle(indexes)
    start_index,next_index=0,BATCH_SIZE

    test_indexes=np.arange(len(test_images))
    random.shuffle(test_indexes)
    test_start_index,test_next_index=0,BATCH_SIZE

    input_images=tf.placeholder(dtype=tf.float32,shape=(None,28,28,1),name='input')
    input_labels=tf.placeholder(dtype=tf.int64,shape=(None),name='labels')

    print(input_images)

    logits,features,total_loss,accuracy,centers_update_op=build_network(input_images=input_images,labels=input_labels)

    optimizer=tf.train.AdamOptimizer(0.001)

    with tf.control_dependencies([centers_update_op]):
        train_op=optimizer.minimize(total_loss,global_step=global_step)

    summary_op=tf.summary.merge_all()

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('./mnist_log_openset',sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    step=sess.run(global_step)
    saver=tf.train.Saver()

    while step<=8000:
        start_index,image_batch,label_batch=get_batch(start_index,indexes,images,labels)
        _, summary_str, train_acc = sess.run(
            [train_op, summary_op, accuracy],feed_dict={input_images:image_batch,input_labels:label_batch})
        step += 1

        writer.add_summary(summary_str, global_step=step)

        if step % 200 == 0:
            test_start_index,test_images_batch,test_labels_batch=get_batch(test_start_index,test_indexes,test_images,test_labels)
            test_acc=sess.run(accuracy,feed_dict={input_images:test_images_batch,input_labels:test_labels_batch})
            print(("step: {}, train_acc:{:.4f}, test_acc{:.4f}".format(step, train_acc,test_acc)))
            saver.save(sess,save_path=model_dir,global_step=step)