from tensorflow import keras
import tensorflow as tf
import os
import numpy as np

model_name = 'face_classifier_VGG16.h5'
face_classifier = keras.models.load_model(f'models/{model_name}')


def test_image_classifier_with_folder(model, path, y_true, img_height=250, img_width=250, class_names=['HugoSilva', 'Unknown']):
    '''
    Read all images from 'path' using tensorflow.keras.preprocessing.image module, 
    than classifies them using 'model' and compare result with 'y_true'.
    Calculate total accuracy based on 'path' test set.

    Parameters:
        model : Image classifier
        path (str): Path to the folder with images you want to test classifier on 
        y_true : True label of the images in the folder. Must be in 'class_names' list
        img_height (int): The height of the image that the classifier can process 
        img_width (int): The width of the image that the classifier can process
        class_names (array-like): List of class names 

    Returns:
        None
    '''
    num_classes = len(class_names)  # Number of classes
    total = 0  # number of images total
    correct = 0  # number of images classified correctly

    for filename in os.listdir(path):
        # read each image in the folder and classifies it
        test_path = os.path.join(path, filename)
        if (".DS_Store" in test_path):
            continue
        print(test_path)
        test_image = keras.preprocessing.image.load_img(
            test_path, target_size=(img_height, img_width, 3))
        # from image to array, can try type(test_image)
        test_image = keras.preprocessing.image.img_to_array(test_image)
        # shape from (250, 250, 3) to (1, 250, 250, 3)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        y_pred = class_names[np.array(result[0]).argmax(
            axis=0)]  # predicted class
        iscorrect = 'correct' if y_pred == y_true else 'incorrect'
        print('{} - {}'.format(iscorrect, filename))
        for index in range(num_classes):
            print("\t{:6} with probabily of {:.2f}%".format(
                class_names[index], result[0][index] * 100))

        total += 1
        if y_pred == y_true:
            correct += 1

    print("\nTotal accuracy is {:.2f}% = {}/{} samples classified correctly".format(
        correct/total*100, correct, total))


test_image_classifier_with_folder(face_classifier,
                                  'datasets/test_images/HugoSilva',
                                  y_true='HugoSilva')

test_image_classifier_with_folder(face_classifier,
                                  'datasets/test_images/Unknown',
                                  y_true='Unknown')
