import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#######################
# train history and save model
#######################

history = model.fit(x_train, y_train, epochs=5)
history_dict = history.history
print("History : %s" % history_dict)
print("History Keys : %s" % history_dict.keys())

result = model.evaluate(x_test, y_test)
print('Evaluate Result:', result)

#######################
# save and load model
#######################

base_path = "../../output"
h5_path = "%s/hello_mnist_1.h5" % base_path
model.save(h5_path)

loaded_model = tf.keras.models.load_model(h5_path)
loaded_model.summary()

#######################
# loaded_model evaluate and predict
#######################

result_loss, result_acc = loaded_model.evaluate(x_test, y_test)
print('Loaded Model Test accuracy:', result_acc)

predictions = loaded_model.predict(x_test)
predictions_labels = [np.argmax(predictions[i]) for i in range(len(x_test))]

index = np.arange(0, len(y_test))
diff = index[predictions_labels != y_test]
print("%d differences: \n%s" % (len(diff), diff))
