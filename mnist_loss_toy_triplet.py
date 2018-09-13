#coding=utf-8
import tensorflow as tf
import os
import tflearn
import random
import numpy as np
import PIL.Image as Image
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import data_flow_ops
from six.moves import xrange
import itertools

#softmax loss 与 center loss的比重调节
LAMBDA=0.5
#center的更新权重
CENTER_LOSS_ALPHA=0.5
NUM_CLASSES=9
model_dir='./models/model-openset-triplet'
test_data_file_path='./mnist/mnist_test'
data_file_path='./mnist/mnist_train'
BATCH_SIZE=128
slim=tf.contrib.slim
ALPHA=0.2
CLASS_PER_BATCH=8
IMAGE_PER_CLASS=100

#--------gpu use only----------
os.environ['CUDA_VISIBLE_DEVICES']='0'

global_step=tf.Variable(0,trainable=False,name='global_step')

def get_triplet_loss(anchor,positive,negative,alpha):
    pos_dist=tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),1)
    neg_dist=tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),1)

    basic_loss=tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    loss=tf.reduce_mean(tf.maximum(basic_loss,0.0),0)
    return loss

def inference(input_images):
    with tf.variable_scope(name_or_scope='',reuse=tf.AUTO_REUSE ):
        x0=tf.layers.conv2d(inputs=input_images,filters=32,kernel_size=3,padding='SAME',name='conv1_1')
        x1 = tf.layers.conv2d(inputs=x0, filters=32, kernel_size=3, padding='SAME',name='conv1_2')
        x2=tf.layers.max_pooling2d(inputs=x1,pool_size=2,strides=2,name='pool1')

        x3=tf.layers.conv2d(inputs=x2,filters=64,kernel_size=3,padding='SAME',name='conv2_1')
        x4 = tf.layers.conv2d(inputs=x3, filters=64, kernel_size=3, padding='SAME',name='conv2_2')
        x5=tf.layers.max_pooling2d(inputs=x4,pool_size=2,strides=2,name='pool2')

        x6=tf.layers.conv2d(inputs=x5,filters=128,kernel_size=3,padding='SAME',name='conv3_1')
        x7 = tf.layers.conv2d(inputs=x6, filters=128, kernel_size=3, padding='SAME',name='conv3_2')
        x8=tf.layers.max_pooling2d(inputs=x7,pool_size=2,strides=2,name='pool3')

        x9=tf.layers.flatten(inputs=x8,name='flatten')

        feature=tf.layers.dense(inputs=x9,units=2,name='fc1')
    #x=tflearn.prelu(feature)
    print(feature)
    #x=tf.layers.dense(inputs=x,units=NUM_CLASSES,name='fc2')
    return feature

def get_image_path(file_path,RANGE=NUM_CLASSES,ratio=1.0):
    images_path=[]
    for i in range(RANGE):
        tmp = os.path.join(file_path, str(i))
        tmp_image_path=[]
        for _root, _dirs, _paths in os.walk(tmp):
            for _path in _paths:
                if random.random()>ratio:continue
                temp=os.path.join(tmp,_path)
                tmp_image_path.append(temp)
            images_path.append(tmp_image_path)
    return images_path


def sample_class(dataset,class_per_batch,image_per_class):
    nrof_images=class_per_batch*image_per_class
    nrof_classes=NUM_CLASSES
    class_indices=np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i=0
    image_paths=[]
    num_per_class=[]
    label_batch=[]
    while len(image_paths)<nrof_images:
        class_index=class_indices[i]
        nrof_images_in_class=len(dataset[class_index])

        image_indices=np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class=min(nrof_images_in_class,image_per_class,nrof_images-len(image_paths))
        idx=image_indices[0:nrof_images_from_class]
        image_paths_for_class=[dataset[class_index][j]for j in idx]
        image_paths+=image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        label_batch+=[class_index for j in idx]
        i+=1
    return image_paths,num_per_class,label_batch

def select_triplets(embeddings,nrof_images_per_class,labels,image_paths,class_per_batch,alpha):
    trip_idx=0
    emb_start_idx=0
    num_trips=0
    triplets=[]
    triplets_label=[]

    for i in xrange(class_per_batch):
        nrof_images=int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            #选择第一张照片作为anchor
            a_idx=emb_start_idx+j-1
            #其他所有图离anchor的距离
            neg_dists_sqr=np.sum(np.square(embeddings[a_idx]-embeddings),1)
            #与之匹配的positive的项
            for pair in xrange(j,nrof_images):
                p_idx=emb_start_idx+pair
                pos_dist_sqr=np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images]=alpha
                all_neg=np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0]
                nrof_random_negs=all_neg.shape[0]

                if nrof_random_negs>0:
                    rnd_idx=np.random.randint(nrof_random_negs)
                    n_idx=all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx],image_paths[p_idx],image_paths[n_idx]))
                    triplets_label.append((labels[a_idx],labels[p_idx],labels[n_idx]))
                    trip_idx+=1

                num_trips+=1
        emb_start_idx+=nrof_images

    np.random.shuffle(triplets)
    return triplets,num_trips,len(triplets),triplets_label

def get_image_pair(image_paths):
    images_anc=[]
    images_pos=[]
    images_neg=[]
    for pair in image_paths:
        img = Image.open(pair[0])
        img_np = np.array(img)
        img_np = img_np.reshape((28, 28, 1))
        images_anc.append(img_np)
        img = Image.open(pair[1])
        img_np = np.array(img)
        img_np = img_np.reshape((28, 28, 1))
        images_pos.append(img_np)
        img = Image.open(pair[2])
        img_np = np.array(img)
        img_np = img_np.reshape((28, 28, 1))
        images_neg.append(img_np)
    images=np.array(images_anc+images_pos+images_neg)
    mean_data=np.mean(images,axis=0)
    images=(images-mean_data)/255.0
    return images

def get_image(image_paths):
    images=[]
    for path in image_paths:
        img=Image.open(path)
        img_np = np.array(img)
        img_np = img_np.reshape((28, 28, 1))
        images.append(img_np)
    images = np.array(images)
    mean_data = np.mean(images, axis=0)
    images = (images - mean_data) / 255.0
    return images

def get_batch(start_index,indexes,images,labels):
    image_batch=[]
    label_batch=[]
    for i in range(BATCH_SIZE):
        image_batch.append(images[indexes[(start_index+i)%len(indexes)]])
        label_batch.append(labels[indexes[(start_index+i)%len(indexes)]])
    return (start_index+BATCH_SIZE)%len(indexes),image_batch,label_batch

#----------------------main-----------------------

if __name__=='__main__':
    dataset=get_image_path(data_file_path)

    input_images=tf.placeholder(dtype=tf.float32,shape=(None,28,28,1),name='input')

    #input_labels=tf.placeholder(dtype=tf.int64,shape=(None),name='labels')

    features = inference(input_images)

    #embedding=tf.nn.l2_normalize(features,1,1e-10,name='embedding')

    [anchor, positive, negative] = tf.split(features,num_or_size_splits=3,axis=0)

    with tf.name_scope('loss'):
        '''
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_labels, logits=prelogits))
        '''
        with tf.name_scope('regularization_loss'):
            regular_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        with tf.name_scope('triplet_loss'):
            triplet_loss=get_triplet_loss(anchor,positive,negative,ALPHA)
        with tf.name_scope('total_loss'):
            #total_loss=tf.add(tf.add_n([triplet_loss],regular_loss),softmax_loss)
            total_loss=tf.add_n([triplet_loss],regular_loss)
    '''        
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(prelogits, 1), input_labels), tf.float32))
    with tf.name_scope('loss/'):
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
    '''

    opt=tf.train.AdagradOptimizer(0.001)
    grads=opt.compute_gradients(total_loss,tf.global_variables())
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(
        0.99, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    #summary_op=tf.summary.merge_all()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./mnist_log_openset', sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    step = sess.run(global_step)
    saver = tf.train.Saver()
    while step<80000:
        image_paths,num_per_class,label_batch=sample_class(dataset,CLASS_PER_BATCH,IMAGE_PER_CLASS)
        images_batch=get_image(image_paths)
        nrof_examples=CLASS_PER_BATCH*IMAGE_PER_CLASS

        nrof_batchs=int(np.ceil(nrof_examples/BATCH_SIZE))

        embeddings = sess.run(features, feed_dict={input_images: images_batch})
        emb_array=np.array(embeddings)
        triplets,nrof_random_negs,nrof_triplets,tri_labels=select_triplets(emb_array,num_per_class,label_batch,image_paths,
                                                                    CLASS_PER_BATCH,ALPHA)
        print(nrof_triplets)
        nrof_batches=int(np.ceil(nrof_triplets*3/BATCH_SIZE))
        triplet_paths=list(itertools.chain(*triplets))
        tri_labels=list(itertools.chain(*tri_labels))
        labels_array=np.reshape(tri_labels,(-1,3))
        triplet_paths_array=np.reshape(np.expand_dims(np.array(triplet_paths),1),(-1,3))

        indices=np.arange(triplet_paths_array.shape[0])
        random.shuffle(indices)

        start_index=0

        nrof_examples=len(triplet_paths)
        i=0
        while i <nrof_batches:
            if start_index+BATCH_SIZE<len(indices):
                img_paths=triplet_paths_array[indices[start_index:start_index+BATCH_SIZE]]
                labels=labels_array[indices[start_index:start_index+BATCH_SIZE]]
            else:
                indices_range=np.concatenate((indices[start_index:],indices[:BATCH_SIZE-(len(indices)-start_index)]),axis=0)
                img_paths=triplet_paths_array[indices_range]
                labels=labels_array[indices_range]
            start_index=(start_index+BATCH_SIZE)%len(indices)
            images=get_image_pair(img_paths)

            err,_=sess.run([total_loss,train_op],feed_dict={input_images:images})

            if step%50==0:
                print('Batch: [%d]\t loss:(%2.3f)' %( step + 1,err))


            step+=1

            if step%200==0:
                saver.save(sess, save_path=model_dir, global_step=step)


