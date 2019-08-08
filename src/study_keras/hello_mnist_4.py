import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

print("y_train \n%s" % y_train[:10])
print("y_train_one_hot \n%s" % y_train_one_hot[:10])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_one_hot, epochs=5)

result_loss, result_acc = model.evaluate(x_test, y_test_one_hot)
print('Test accuracy:', result_acc)

predictions = model.predict(x_test)
predictions_labels = [np.argmax(predictions[i]) for i in range(len(x_test))]

index = np.arange(0, len(y_test))
diff = index[predictions_labels != y_test]
print("%d differences: \n%s" % (len(diff), diff))
