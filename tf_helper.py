import tensorflow as tf
from tensorflow_vgg import vgg16_avg_pool

def compute_tf_output(img_array):
    tf.reset_default_graph()
    
    vgg = vgg16_avg_pool.Vgg16()
    vgg.build(img_array)
    content_layers_list = dict({0: vgg.conv1_1, 1: vgg.conv1_2, 2: vgg.pool1, 3: vgg.conv2_1, 4: vgg.conv2_2, 5: vgg.pool2, 6: vgg.conv3_1, 7: vgg.conv3_2, 8: vgg.conv3_3, 9: vgg.pool3, 10: vgg.conv4_1, 11: vgg.conv4_2, 12: vgg.conv4_3, 13: vgg.pool4, 14: vgg.conv5_1, 15: vgg.conv5_2, 16: vgg.conv5_3, 17: vgg.pool5 })

    img_layer_outputs = dict()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tf.trainable_variables())
        for i in range(len(content_layers_list)):
            img_layer_outputs[i] = sess.run(content_layers_list[i])
            #print("No. ", i, " ", content_layers_list[i].name, "completed.")
        print("All layers' outputs have been computed sucessfully.")
    
    return img_layer_outputs