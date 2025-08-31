import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models # type: ignore

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Sheep', 'Truck']

model = models.load_model('img_classifier.keras')

img = cv.imread('deer.jpg')
img = cv.resize(img, (32, 32))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.show(img)

prediction = model.predict(np.array([img]) / 255)

index = np.argmax(prediction)

print(f'prediction: {class_names[index]}')