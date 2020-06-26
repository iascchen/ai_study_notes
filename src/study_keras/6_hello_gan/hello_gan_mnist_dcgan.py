import glob
import os
import time

import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.losses as losses

plt.rcParams['figure.max_open_warning'] = 60


def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def generator_loss(generated_output):
    return losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output),
                                             logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output),
                                                  logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


def train_step(generator, discriminator, images):
    # generating noise from a normal distribution
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))


def train(generator, discriminator, dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            train_step(generator, discriminator, images)

        generate_and_save_images(generator,
                                 epoch + 1,
                                 random_vector_for_generation)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                         time.time() - start))
    # generating after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             random_vector_for_generation)


def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('dcgan_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


if __name__ == '__main__':
    tf.enable_eager_execution()
    print("Eager execution: {}".format(tf.executing_eagerly()))

    image_path = "../../../images"
    output_path = "../../../output"

    gen_img_prefix = '%s/paint_chinese' % output_path

    (train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    _generator = make_generator_model()
    _discriminator = make_discriminator_model()

    generator_optimizer = tf.train.AdamOptimizer(1e-4)
    discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

    checkpoint_dir = '%s/training_checkpoints' % output_path
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=_generator,
                                     discriminator=_discriminator)

    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    # We'll re-use this random vector used to seed the generator so
    # it will be easier to see the improvement over time.
    random_vector_for_generation = tf.random.normal([num_examples_to_generate,
                                                     noise_dim])

    # train_step = tf.contrib.eager.defun(train_step)
    train(_generator, _discriminator, train_dataset, EPOCHS)

    with imageio.get_writer('dcgan.gif', mode='I') as writer:
        filenames = glob.glob('dcgan*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    # this is a hack to display the gif inside the notebook
    os.system('cp dcgan.gif dcgan.gif.png')
