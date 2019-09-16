import datetime
import json
import time
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from imageio import imwrite, imsave
from style_transfer_utils import get_style_loss, get_content_loss, get_tv_loss, residual_block, OutputScale, \
    InputReflect, AverageAddTwo, process_image, expand_input, get_vgg_activation, dummy_loss, zero_loss, \
    deprocess_image, get_padding, remove_padding
from tensorflow.keras import optimizers
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.python.keras import Model
from tensorflow.python.layers import layers

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


def get_training_model(width, height, bs=1, bi_style=False):
    input_o = layers.Input(shape=(height, width, 3), dtype='float32', name='input_o')

    c1 = layers.Conv2D(32, (9, 9), strides=1, padding='same', name='conv_1')(input_o)
    c1 = layers.BatchNormalization(name='normal_1')(c1)
    c1 = layers.Activation('relu', name='relu_1')(c1)

    c2 = layers.Conv2D(64, (3, 3), strides=2, padding='same', name='conv_2')(c1)
    c2 = layers.BatchNormalization(name='normal_2')(c2)
    c2 = layers.Activation('relu', name='relu_2')(c2)

    c3 = layers.Conv2D(128, (3, 3), strides=2, padding='same', name='conv_3')(c2)
    c3 = layers.BatchNormalization(name='normal_3')(c3)
    c3 = layers.Activation('relu', name='relu_3')(c3)

    r1 = residual_block(c3, 1)
    r2 = residual_block(r1, 2)
    r3 = residual_block(r2, 3)
    r4 = residual_block(r3, 4)
    r5 = residual_block(r4, 5)

    d1 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', name='conv_4')(r5)
    d1 = layers.BatchNormalization(name='normal_4')(d1)
    d1 = layers.Activation('relu', name='relu_4')(d1)

    d2 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', name='conv_5')(d1)
    d2 = layers.BatchNormalization(name='normal_5')(d2)
    d2 = layers.Activation('relu', name='relu_5')(d2)

    c4 = layers.Conv2D(3, (9, 9), strides=1, padding='same', name='conv_6')(d2)
    c4 = layers.BatchNormalization(name='normal_6')(c4)
    c4 = layers.Activation('tanh', name='tanh_1')(c4)
    c4 = OutputScale(name='output')(c4)

    content_activation = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
    style_activation1 = layers.Input(shape=(height, width, 64), dtype='float32')
    style_activation2 = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
    style_activation3 = layers.Input(shape=(height // 4, width // 4, 256), dtype='float32')
    style_activation4 = layers.Input(shape=(height // 8, width // 8, 512), dtype='float32')

    if bi_style:
        style_activation1_2 = layers.Input(shape=(height, width, 64), dtype='float32')
        style_activation2_2 = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
        style_activation3_2 = layers.Input(shape=(height // 4, width // 4, 256), dtype='float32')
        style_activation4_2 = layers.Input(shape=(height // 8, width // 8, 512), dtype='float32')

    total_variation_loss = layers.Lambda(get_tv_loss, output_shape=(1,), name='tv',
                                         arguments={'width': width, 'height': height})([c4])

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(c4)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    style_loss1 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style1', arguments={'batch_size': bs})([x, style_activation1])
    if bi_style:
        style_loss1_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style1_2', arguments={'batch_size': bs})([x, style_activation1_2])
        style_loss1 = AverageAddTwo(name='style1_out')([style_loss1, style_loss1_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    content_loss = layers.Lambda(get_content_loss, output_shape=(1,), name='content')([x, content_activation])
    style_loss2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style2', arguments={'batch_size': bs})([x, style_activation2])
    if bi_style:
        style_loss2_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style2_2', arguments={'batch_size': bs})([x, style_activation2_2])
        style_loss2 = AverageAddTwo(name='style2_out')([style_loss2, style_loss2_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    style_loss3 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style3', arguments={'batch_size': bs})([x, style_activation3])
    if bi_style:
        style_loss3_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style3_2', arguments={'batch_size': bs})([x, style_activation3_2])
        style_loss3 = AverageAddTwo(name='style3_out')([style_loss3, style_loss3_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    style_loss4 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style4', arguments={'batch_size': bs})([x, style_activation4])
    if bi_style:
        style_loss4_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style4_2', arguments={'batch_size': bs})([x, style_activation4_2])
        style_loss4 = AverageAddTwo(name='style4_out')([style_loss4, style_loss4_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if bi_style:
        model = Model(
            [input_o, content_activation, style_activation1, style_activation2, style_activation3, style_activation4,
             style_activation1_2, style_activation2_2, style_activation3_2, style_activation4_2],
            [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, c4])
    else:
        model = Model(
            [input_o, content_activation, style_activation1, style_activation2, style_activation3, style_activation4],
            [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, c4])
    model_layers = {layer.name: layer for layer in model.layers}
    original_vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    original_vgg_layers = {layer.name: layer for layer in original_vgg.layers}

    # load image_net weight
    for layer in original_vgg.layers:
        if layer.name in model_layers:
            model_layers[layer.name].set_weights(original_vgg_layers[layer.name].get_weights())
            model_layers[layer.name].trainable = False

    print("training model built successfully!")
    return model


def get_evaluate_model(width, height):
    input_o = layers.Input(shape=(height, width, 3), dtype='float32', name='input_o')

    c1 = layers.Conv2D(32, (9, 9), strides=1, padding='same', name='conv_1')(input_o)
    c1 = layers.BatchNormalization(name='normal_1')(c1)
    c1 = layers.Activation('relu', name='relu_1')(c1)

    c2 = layers.Conv2D(64, (3, 3), strides=2, padding='same', name='conv_2')(c1)
    c2 = layers.BatchNormalization(name='normal_2')(c2)
    c2 = layers.Activation('relu', name='relu_2')(c2)

    c3 = layers.Conv2D(128, (3, 3), strides=2, padding='same', name='conv_3')(c2)
    c3 = layers.BatchNormalization(name='normal_3')(c3)
    c3 = layers.Activation('relu', name='relu_3')(c3)

    r1 = residual_block(c3, 1)
    r2 = residual_block(r1, 2)
    r3 = residual_block(r2, 3)
    r4 = residual_block(r3, 4)
    r5 = residual_block(r4, 5)

    d1 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', name='conv_4')(r5)
    d1 = layers.BatchNormalization(name='normal_4')(d1)
    d1 = layers.Activation('relu', name='relu_4')(d1)

    d2 = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', name='conv_5')(d1)
    d2 = layers.BatchNormalization(name='normal_5')(d2)
    d2 = layers.Activation('relu', name='relu_5')(d2)

    c4 = layers.Conv2D(3, (9, 9), strides=1, padding='same', name='conv_6')(d2)
    c4 = layers.BatchNormalization(name='normal_6')(c4)
    c4 = layers.Activation('tanh', name='tanh_1')(c4)
    c4 = OutputScale(name='output')(c4)

    model = Model([input_o], c4)
    print("evaluate model built successfully!")
    return model


def get_temp_view_model(width, height, bs=1, bi_style=False):
    input_o = layers.Input(shape=(height, width, 3), dtype='float32')

    y = InputReflect(width, height, name='output')(input_o)
    total_variation_loss = layers.Lambda(get_tv_loss, output_shape=(1,), name='tv',
                                         arguments={'width': width, 'height': height})([y])

    content_activation = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
    style_activation1 = layers.Input(shape=(height, width, 64), dtype='float32')
    style_activation2 = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
    style_activation3 = layers.Input(shape=(height // 4, width // 4, 256), dtype='float32')
    style_activation4 = layers.Input(shape=(height // 8, width // 8, 512), dtype='float32')

    if bi_style:
        style_activation1_2 = layers.Input(shape=(height, width, 64), dtype='float32')
        style_activation2_2 = layers.Input(shape=(height // 2, width // 2, 128), dtype='float32')
        style_activation3_2 = layers.Input(shape=(height // 4, width // 4, 256), dtype='float32')
        style_activation4_2 = layers.Input(shape=(height // 8, width // 8, 512), dtype='float32')

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(y)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    style_loss1 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style1', arguments={'batch_size': bs})([x, style_activation1])
    if bi_style:
        style_loss1_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style1_2', arguments={'batch_size': bs})([x, style_activation1_2])
        style_loss1 = AverageAddTwo(name='style1_out')([style_loss1, style_loss1_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    content_loss = layers.Lambda(get_content_loss, output_shape=(1,), name='content')([x, content_activation])
    style_loss2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style2', arguments={'batch_size': bs})([x, style_activation2])
    if bi_style:
        style_loss2_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style2_2', arguments={'batch_size': bs})([x, style_activation2_2])
        style_loss2 = AverageAddTwo(name='style2_out')([style_loss2, style_loss2_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    style_loss3 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style3', arguments={'batch_size': bs})([x, style_activation3])
    if bi_style:
        style_loss3_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style3_2', arguments={'batch_size': bs})([x, style_activation3_2])
        style_loss3 = AverageAddTwo(name='style3_out')([style_loss3, style_loss3_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    style_loss4 = layers.Lambda(get_style_loss, output_shape=(1,),
                                name='style4', arguments={'batch_size': bs})([x, style_activation4])
    if bi_style:
        style_loss4_2 = layers.Lambda(get_style_loss, output_shape=(1,),
                                      name='style4_2', arguments={'batch_size': bs})([x, style_activation4_2])
        style_loss4 = AverageAddTwo(name='style4_out')([style_loss4, style_loss4_2])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if bi_style:
        model = Model(
            [input_o, content_activation, style_activation1, style_activation2, style_activation3,
             style_activation4,
             style_activation1_2, style_activation2_2, style_activation3_2, style_activation4_2],
            [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, y])
    else:
        model = Model(
            [input_o, content_activation, style_activation1, style_activation2, style_activation3,
             style_activation4],
            [content_loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, y])
    model_layers = {layer.name: layer for layer in model.layers}
    original_vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    original_vgg_layers = {layer.name: layer for layer in original_vgg.layers}

    # load image_net weight
    for layer in original_vgg.layers:
        if layer.name in model_layers:
            model_layers[layer.name].set_weights(original_vgg_layers[layer.name].get_weights())
            model_layers[layer.name].trainable = False

    print("temp_view model built successfully!")
    return model


def train(options):
    width = options["train_image_width"]
    height = options["train_image_height"]

    # Get style activations
    style_tensor = process_image(options["style_image_path"], width, height)
    style_acts = list()
    for layer_name in options["style_layer"]:
        func = get_vgg_activation(layer_name, width, height)
        style_act = expand_input(options["batch_size"], func([style_tensor])[0])
        style_acts.append(style_act)

    if "style_image_path_2" in options:
        style_tensor_2 = process_image(options["style_image_path_2"], width, height)
        style_acts_2 = list()
        for layer_name in options["style_layer"]:
            func = get_vgg_activation(layer_name, width, height)
            style_act_2 = expand_input(options["batch_size"], func([style_tensor_2])[0])
            style_acts_2.append(style_act_2)

    # Get content activations for test_image
    content_test = process_image(options["test_image_path"], width, height)
    content_func = get_vgg_activation(options["content_layer"], width, height)
    content_act_test = expand_input(options["batch_size"], content_func([content_test])[0])
    content_test = expand_input(options["batch_size"], content_test)

    # Get weights
    style_w = options["style_weight"] / len(style_acts)
    content_w = options["content_weight"]
    tv_w = options["total_variation_weight"]

    # Get training model
    bi_style = False
    if "style_image_path_2" in options:
        bi_style = True
    training_model = get_training_model(width, height, bs=options['batch_size'], bi_style=bi_style)
    if bi_style:
        training_model.compile(loss={'content': dummy_loss, 'style1_out': dummy_loss, 'style2_out': dummy_loss,
                                     'style3_out': dummy_loss, 'style4_out': dummy_loss, 'tv': dummy_loss,
                                     'output': zero_loss},
                               optimizer=optimizers.Adam(lr=options["learning_rate"]),
                               loss_weights=[content_w, style_w, style_w, style_w, style_w, tv_w, 0])
    else:
        training_model.compile(loss={'content': dummy_loss, 'style1': dummy_loss, 'style2': dummy_loss,
                                     'style3': dummy_loss, 'style4': dummy_loss, 'tv': dummy_loss, 'output': zero_loss},
                               optimizer=optimizers.Adam(lr=options["learning_rate"]),
                               loss_weights=[content_w, style_w, style_w, style_w, style_w, tv_w, 0])

    # If flag is set, print model summary and generate model description
    if options["plot_model"]:
        training_model.summary()
        plot_model(training_model, to_file='model.png')

    # function for printing test information
    def print_test_results(cur_res, cur_iter, prev_loss):
        losses = list()
        losses.append(cur_res[0][0] * content_w)
        losses.append(cur_res[1][0] * style_w)
        losses.append(cur_res[2][0] * style_w)
        losses.append(cur_res[3][0] * style_w)
        losses.append(cur_res[4][0] * style_w)
        losses.append(cur_res[5][0] * tv_w)
        cur_loss = sum(losses)
        if prev_loss is None:
            prev_loss = cur_loss

        print("----------------------------------------------------")
        print("Details: iteration %d, " % cur_iter, end='')
        print('improvement: %.2f percent, ' % ((prev_loss - cur_loss) / prev_loss * 100), end='')
        print("loss: %.0f" % cur_loss)
        print("content_loss: %.0f, style_loss_1: %.0f, style_loss_2: %.0f\n"
              "style_loss_3: %.0f, style_loss_4: %.0f, tv_loss: %.0f"
              % (losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]))
        print("----------------------------------------------------")

        return cur_loss

    # Prepare for training
    dg = ImageDataGenerator()
    dummy_in = expand_input(options["batch_size"], np.array([0.0]))
    interrupted = False
    c_loss = None
    t_sum = 0.0

    # Begin Training
    t_total_1 = time.time()
    for i in range(options["epochs"]):
        print("Epoch: %d" % (i + 1))
        iters = 0

        for x in dg.flow_from_directory(options["train_image_path"], class_mode=None,
                                        batch_size=options["batch_size"], target_size=(height, width)):
            try:
                t1 = time.time()
                x = vgg16.preprocess_input(x)
                content_act = content_func([x])[0]
                if bi_style:
                    res = training_model.fit([x, content_act, style_acts[0], style_acts[1], style_acts[2],
                                              style_acts[3], style_acts_2[0], style_acts_2[1], style_acts_2[2],
                                              style_acts_2[3]], [dummy_in, dummy_in, dummy_in, dummy_in, dummy_in,
                                                                 dummy_in, x],
                                             epochs=1, verbose=0, batch_size=options["batch_size"])
                else:
                    res = training_model.fit([x, content_act, style_acts[0], style_acts[1], style_acts[2],
                                              style_acts[3]], [dummy_in, dummy_in, dummy_in, dummy_in, dummy_in,
                                                               dummy_in, x],
                                             epochs=1, verbose=0, batch_size=options["batch_size"])
                t2 = time.time()
                t_sum += t2 - t1

                iters += 1

                if iters % options["view_iter"] == 0:
                    loss = res.history['loss'][0]
                    est_time = int((options["steps_per_epoch"] * (options["epochs"] - i) - iters)
                                   * (t_sum / options["view_iter"]))
                    print("Iter : %d / %d, Time elapsed: %0.2f seconds, Loss: %.0f, EST: " %
                          (iters, options["steps_per_epoch"], t_sum / options["view_iter"], loss) +
                          str(datetime.timedelta(seconds=est_time)))
                    t_sum = 0.0

                if iters % options["test_iter"] == 0:
                    if bi_style:
                        res = training_model.predict([content_test, content_act_test, style_acts[0], style_acts[1],
                                                      style_acts[2], style_acts[3], style_acts_2[0], style_acts_2[1],
                                                      style_acts_2[2], style_acts_2[3]])
                    else:
                        res = training_model.predict([content_test, content_act_test, style_acts[0], style_acts[1],
                                                      style_acts[2], style_acts[3]])
                    c_loss = print_test_results(res, iters, c_loss)

                    output = deprocess_image(res[6][0], width, height)
                    imsave(options["test_res_save_path"] + '%d_%d_output.jpg' % (i, iters), output)

                if iters >= options["steps_per_epoch"]:
                    break

            except KeyboardInterrupt:
                print("Interrupted, training suspended.")
                interrupted = True
                break

        if interrupted:
            break

    t_total_2 = time.time()
    print("Training ended. Time used: " + str(datetime.timedelta(seconds=int(t_total_2 - t_total_1))))

    # Saving models
    print("Saving models...")
    model_eval = get_evaluate_model(width, height)
    training_model_layers = {layer.name: layer for layer in training_model.layers}
    for layer in model_eval.layers:
        if layer.name in training_model_layers:
            print(layer.name)
            layer.set_weights(training_model_layers[layer.name].get_weights())

    model_eval.save_weights(options["weights_save_path"] + '%s_weights.h5' % options["net_name"])


def temp_view(options, img_read_path, img_write_path, iters):
    width = options["train_image_width"]
    height = options["train_image_height"]

    # Get style activations
    style_tensor = K.variable(process_image(options["style_image_path"], width, height))
    style_acts = list()
    for layer_name in options["style_layer"]:
        func = get_vgg_activation(layer_name, width, height)
        style_act = func([style_tensor])[0]
        style_acts.append(style_act)

    if "style_image_path_2" in options:
        style_tensor_2 = process_image(options["style_image_path_2"], width, height)
        style_acts_2 = list()
        for layer_name in options["style_layer"]:
            func = get_vgg_activation(layer_name, width, height)
            style_act_2 = func([style_tensor_2])[0]
            style_acts_2.append(style_act_2)

    # Get content activations
    content_tensor = K.variable(process_image(img_read_path, width, height))
    func = get_vgg_activation(options["content_layer"], width, height)
    content_act = func([content_tensor])[0]

    dummy_in = np.array([0.0])
    style_w = options["style_weight"] / len(style_acts)
    content_w = options["content_weight"]
    tv_w = options["total_variation_weight"]

    # Get training model
    bi_style = False
    if "style_image_path_2" in options:
        bi_style = True
    training_model = get_temp_view_model(width, height, bi_style=bi_style)
    if bi_style:
        training_model.compile(loss={'content': dummy_loss, 'style1_out': dummy_loss, 'style2_out': dummy_loss,
                                     'style3_out': dummy_loss, 'style4_out': dummy_loss, 'tv': dummy_loss,
                                     'output': zero_loss},
                               optimizer=optimizers.Adam(lr=1),
                               loss_weights=[content_w, style_w, style_w, style_w, style_w, tv_w, 0])
    else:
        training_model.compile(loss={'content': dummy_loss, 'style1': dummy_loss, 'style2': dummy_loss,
                                     'style3': dummy_loss, 'style4': dummy_loss, 'tv': dummy_loss, 'output': zero_loss},
                               optimizer=optimizers.Adam(lr=1),
                               loss_weights=[content_w, style_w, style_w, style_w, style_w, tv_w, 0])

    # If flag is set, print model summary and generate model description
    if options["plot_model"]:
        training_model.summary()
        plot_model(training_model, to_file='model.png')

    # Input should always be ones
    x = np.ones([1, height, width, 3], dtype='float32')

    # Begin training
    prev_loss = None
    for i in range(iters):
        t1 = time.time()

        if bi_style:
            res = training_model.fit(
                [x, content_act, style_acts[0], style_acts[1], style_acts[2], style_acts[3], style_acts_2[0],
                 style_acts_2[1], style_acts_2[2], style_acts_2[3]],
                [dummy_in, dummy_in, dummy_in, dummy_in, dummy_in, dummy_in, x],
                epochs=1, verbose=0, batch_size=1)
        else:
            res = training_model.fit([x, content_act, style_acts[0], style_acts[1], style_acts[2], style_acts[3]],
                                     [dummy_in, dummy_in, dummy_in, dummy_in, dummy_in, dummy_in, x],
                                     epochs=1, verbose=0, batch_size=1)

        t2 = time.time()

        if i % 10 == 0:
            loss = res.history['loss'][0]
            if prev_loss is None:
                prev_loss = loss
            improvement = (prev_loss - loss) / prev_loss * 100
            prev_loss = loss

            print("Iter: %d / %d, Time elapsed: %0.2f seconds, Loss: %.0f, Improvement: %0.2f percent." %
                  (i, iters, t2 - t1, loss, improvement))
            if bi_style:
                print("Detail: content_loss: %0.0f, style_loss_1: %0.0f, style_loss_2: %0.0f,"
                      " style_loss_3: %0.0f, style_loss_4: %0.0f, tv_loss: %0.0f"
                      % (float(res.history['content_loss'][0]) * content_w,
                         float(res.history['style1_out_loss'][0]) * style_w,
                         float(res.history['style2_out_loss'][0]) * style_w,
                         float(res.history['style3_out_loss'][0]) * style_w,
                         float(res.history['style4_out_loss'][0]) * style_w,
                         float(res.history['tv_loss'][0]) * tv_w))
            else:
                print("Detail: content_loss: %0.0f, style_loss_1: %0.0f, style_loss_2: %0.0f,"
                      " style_loss_3: %0.0f, style_loss_4: %0.0f, tv_loss: %0.0f"
                      % (float(res.history['content_loss'][0]) * content_w,
                         float(res.history['style1_loss'][0]) * style_w,
                         float(res.history['style2_loss'][0]) * style_w,
                         float(res.history['style3_loss'][0]) * style_w,
                         float(res.history['style4_loss'][0]) * style_w,
                         float(res.history['tv_loss'][0]) * tv_w))

    if bi_style:
        res = training_model.predict(
            [x, content_act, style_acts[0], style_acts[1], style_acts[2], style_acts[3], style_acts_2[0],
             style_acts_2[1], style_acts_2[2], style_acts_2[3]])
    else:
        res = training_model.predict([x, content_act, style_acts[0], style_acts[1], style_acts[2], style_acts[3]])
    output = deprocess_image(res[6][0], width, height)
    imsave(img_write_path, output)


def predict(options, img_read_path, img_write_path):
    # Read image
    content = process_image(img_read_path, -1, -1, resize=False)
    ori_height = content.shape[1]
    ori_width = content.shape[2]

    # Pad image
    content = get_padding(content)
    height = content.shape[1]
    width = content.shape[2]

    # Get eval model
    eval_model = get_evaluate_model(width, height)
    eval_model.load_weights(options['weights_read_path'])

    # If flag is set, print model summary and generate model description
    if options["plot_model"]:
        eval_model.summary()
        plot_model(eval_model, to_file='model.png')

    # Generate output and save image
    res = eval_model.predict([content])
    output = deprocess_image(res[0], width, height)
    output = remove_padding(output, ori_height, ori_width)
    imwrite(img_write_path, output)


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('-c', type=str, dest='config_path', help='config path',
                        metavar='CONFIG_PATH', required=True)
    parser.add_argument('-m', type=str, dest='mode', help='train, predict or temp_view',
                        metavar='MODE', required=True)
    parser.add_argument('-i', type=str, dest='image_path', help='image for transformation or viewing',
                        metavar='IMAGE_PATH')
    parser.add_argument('-o', type=str, dest='image_output_path', help='image output path',
                        metavar='IMAGE_OUTPUT_PATH')
    parser.add_argument('--iters', type=int, dest='iters', help='iter times, only for temp_view mode',
                        metavar='ITER_TIMES', default=500)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    with open(args.config_path) as f_config:
        options = json.load(f_config)

    if args.mode == 'train':
        train(options)
    elif args.mode == 'predict':
        predict(options, args.image_path, args.image_output_path)
    elif args.mode == 'temp_view':
        temp_view(options, args.image_path, args.image_output_path, args.iters)
