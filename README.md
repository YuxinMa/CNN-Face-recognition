# CNN-Face-recognition
Python (sklearn, keras)


For face detection, Haar-Cascades were used and for face recognition.

Used Python (OpenCV+keras) to apply Convolutional Neural Network (CNN) model to conduct face recognition from CVL Face Database (114 persons, 7 images for each person, resolution: 640*480 pixels)

Applied PCA for dimensionality reduction; applied SVM and RF for classification to make comparison (Required)

Constructed Xception and VGGNet-16 model and trained with SGD+momentum optimizer. Defined data generator for data augmentation. Evaluated and compared two models with the higher accuracy of 97.1%

Xception:
Selecting 4,692 images as the training set (after data augmentation), 586 images as the validation set, and 587 images as the training set. A total of 300 epochs were trained, and finally the classification accuracy was 97.10%.

VGGNet-16:
Selecting 4,692 images as the training set (after data augmentation), 586 images as the validation set, and 587 images as the training set. A total of 300 epochs were trained, and finally the classification accuracy was 32.20%.
