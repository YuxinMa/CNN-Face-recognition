# CNN-Face-recognition
Python (sklearn, keras)


For face detection, Haar-Cascades were used and for face recognition.

Used Python (OpenCV+keras) to apply Convolutional Neural Network (CNN) model to conduct face recognition from CVL Face Database (114 persons, 7 images for each person, resolution: 640*480 pixels)

Applied PCA for dimensionality reduction; applied SVM and RF for classification to make comparison (Required)

Constructed VGGNet-16 model with dropout layer preventing overfitting and trained with SGD+momentum optimizer. Defined data generator for data augmentation. Evaluated the model with the accuracy of 88%
