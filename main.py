import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import cv2 as cv
import numpy as np
from keras.models import load_model

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
# plt.show()


training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]


# # Train the model using a convultional neural network
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images,testing_labels))


# loss, accuracy = model.evaluate(testing_images, testing_labels)

# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# # model.save('image_classifier.h5')
# model.save('image_classifier.keras')

model = models.load_model('image_classifier.keras')

# Images need to be sclaed down to the size of the trained images (The size of the first Convultion layer)

# Use opencv adn numpy to load the images into a script, make a prediction, and print the image to the screen
# OpenCV read images as RBG instead of RGB so they need to be converted so Matplotlib and we trainged the data on RGB schema

img = cv.imread('plane.jpeg')

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

# pass the image in a numpy array
prediction = model.predict(np.array([img])/255)
plt.show()
# With the softmax activation function we receive a likelihood of each classification. We need to take the highest number from the array
index = np.argmax(prediction)

print(f'CLassname: {class_names[index]}')