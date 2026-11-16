import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image

#Load the pretrained model
model = MobileNetV2(weights='imagenet')

print("Pretrained model loaded successfully!")

#Load an image
img = image.load_img(
'cat.jpeg',
target_size=(224, 224)
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

img_array = preprocess_input(img_array)

print("Image loaded successfully!")

#Predict the image class
predictions = model.predict(img_array)

results = decode_predictions(predictions, top=3)

for _, label, probability in results[0]:
    print(label, ":", round(float(probability) * 100, 2), "%")

#Visualize the input image
import matplotlib.pyplot as plt

plt.imshow(img)
plt.axis('off')
plt.title("Input Image")
plt.show()
