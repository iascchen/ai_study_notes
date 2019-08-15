import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image

if __name__ == '__main__':
    base_path = "../../data"
    output_path = "../../output"
    images_dir = "%s/flower_photos" % base_path
    images_dir_2 = "%s/pets/images" % base_path

    #######################
    # load model
    #######################
    h5_path = "%s/mobilenetv2_transfer_seq.h5" % output_path
    # h5_path = "%s/mobilenetv2_transfer_model.h5" % output_path

    loaded_model = keras.models.load_model(h5_path)


    def process_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        pImg = mobilenet_v2.preprocess_input(img_array)
        return pImg


    # process the test image
    # test_img_path = '%s/Abyssinian_1.jpg' % images_dir_2
    # pImg = process_image(test_img_path)
    #
    # predictions = loaded_model.predict(pImg)
    # predictions_class = np.argmax(predictions)
    # print(predictions_class)

    test_img_path = '%s/roses/99383371_37a5ac12a3_n.jpg' % images_dir
    pImg = process_image(test_img_path)

    predictions = loaded_model.predict(pImg)
    print(predictions)
    predictions_class = np.argmax(predictions)
    print(predictions_class)
