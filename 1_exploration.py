import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Open the TFRecord file for reading
record_iterator = tf.compat.v1.io.tf_record_iterator(path='train_tfrecords/ld_train00-1338.tfrec')

# Iterate over each record in the TFRecord file
for string_record in record_iterator:
    # Parse the serialized Example proto
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # Extract the image data from the Example proto
    image_bytes = example.features.feature['image'].bytes_list.value[0]

    # Convert the image bytes to a NumPy array
    image = np.frombuffer(image_bytes, dtype=np.uint8)

    # Reshape the NumPy array to the original image shape
    image = image.reshape((600, 800, 3))

# Display the image using matplotlib
plt.imshow(image)
plt.show()

# Save the image in JPEG format
plt.imsave('path/to/save/image.jpg', image)


# ##############################################################################################

# image = plt.imread('train_images/1924914.jpg')

# print(image.shape)