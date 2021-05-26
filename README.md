# Introduction
In May 2018 a 31-year-old man identified as "Mr. Ao" was being sought out by the Chinese Communist Party (CCP) on the charge of having committed "economic crimes", because of a facial recognition system the man was located and arrested in a crowd of 60,000 spectator at a concert by the singer Jacky Cheung from Hong Kong.

And this repository is about the technology that located "Mr. Ao".

![recognizer](https://user-images.githubusercontent.com/82250641/119619745-ad1fe180-bdda-11eb-8d2e-e037d1e8a46b.png)
illustrative photo of a facial recognition system.

# History
Automated facial recognition was pioneered in the 1960s. Woody Bledsoe, Helen Chan Wolf, and Charles Bisson worked on using the computer to recognize human faces. Their early facial recognition project was called "man-machine" because the coordinates of the facial features in a photograph had to be established by a human before they could be used by the computer for recognition. 

In 1970, Takeo Kanade publicly demonstrated a face matching system that located anatomical features such as the chin and calculated the distance ratio between facial features without human intervention. Later tests revealed that the system could not always reliably identify facial features.

In 1993, the Defense Advanced Research Project Agency (DARPA) and the Army Research Laboratory (ARL) established the face recognition technology program FERET to develop "automatic face recognition capabilities" that could be employed in a productive real life environment "to assist security, intelligence, and law enforcement personnel in the performance of their duties." Face recognition systems that had been trialed in research labs were evaluated and the FERET tests found that while the performance of existing automated facial recognition systems varied, a handful of existing methods could viably be used to recognize faces in still images taken in a controlled environment.

Following the 1993 FERET face recognition vendor test the Department of Motor Vehicles (DMV) offices in West Virginia and New Mexico were the first DMV offices to use automated facial recognition systems as a way to prevent and detect people obtaining multiple driving licenses under different names.

Until the 1990s facial recognition systems were developed primarily by using photographic portraits of human faces. Research on face recognition to reliably locate a face in an image that contains other objects gained traction in the early 1990s with the principle component analysis (PCA). The PCA method of face detection is also known as Eigenface and was developed by Matthew Turk and Alex Pentland. Turk and Pentland combined the conceptual approach of the Karhunen–Loève theorem and factor analysis, to develop a linear model. 

Real-time face detection in video footage became possible in 2001 with the Viola–Jones object detection framework for faces. Paul Viola and Michael Jones combined their face detection method with the Haar-like feature approach to object recognition in digital images to launch AdaBoost, the first real-time frontal-view face detector. By 2015 the Viola-Jones algorithm had been implemented using small low power detectors on handheld devices and embedded systems. Therefore, the Viola-Jones algorithm has not only broadened the practical application of face recognition systems but has also been used to support new features in user interfaces and teleconferencing.


# How does a facial recognition system work?
 **A facial recognition system can be divided into three main categories:**

1. Face Detection
2. Feature extraction
3. Feature Matching

## 1. Face detection

Face detection is a computer technology that identifies human faces in digital images, extract the faces, and display it (or create a compressed file to use it further for feature extraction).

Methods used in Face Detection:
 
1. Haar cascade Face Detection
2. Dlib (HOG) Face Detection
3. Dlib (CNN) Face Detection
4. MTCNN Face Detection

**Haar cascade Face Detection**

This method has a simple architecture that works nearly real-time on CPU. Also, it can detect images at different scales. But the major drawback is that it gives false results as well as it doesn’t work on non-frontal images.

**Dlib (HOG) Face Detection**

It is the fastest method on CPU which can work on frontal and slightly no-frontal images. But it is incapable of detecting small images and handling occlusions. Also, it often excludes some parts of the chin and forehead while detection.

**Dlib (CNN) Face Detection**

It works very fast on GPU and is capable to work for various face orientations in images. It can also handle occlusions. But the major disadvantage is that it is trained on a minimum face size of 80*80 so it can’t detect small faces in images. It is also very slow on the CPU.

**MTCNN Face Detection**

This method gives the most accurate results out of all the four methods. It works for faces having various orientations in images and can detect faces across various scales. It can even handle occlusions. It doesn’t hold any major drawback as such but is comparatively slower than HOG and Haar cascade method.
This will be the method used in this project

## 2. Feature Extraction

Feature extraction is the basic and most important initializing step for face recognition. It extracts the biological components of your face. These biological components are the features of your face which differ from person to person. There are various methods which extract various combination of features, commonly known as nodal points. No two people can have all the nodal points similar to each other except for identical twins.

Facial Feature extraction has two approaches:
1. Shallow Approach
- PCA (Dimensionality Reduction Method)
- LDA (Dimensionality Reduction Method)
- Cosine Similarity (Feature Matching)
- HOG
- SIFT
2. Deep Approach (Industrial Giant Nets)
- VGG
- Face Recognition API
- FaceNet Keras

**PCA**

PCA is used to reduce the dimensionality of the data. In PCA, the original features of your dataset will be converted into a linear combination of uncorrelated variables(features). These combinations are known as Principal Components. PCA increases algorithm performance and improves visualization. But for reading the PCA components, you need to standardize the data otherwise it won’t be possible to find the optimal Principal Components. Also, PCA results in loss of information if the number of Principal Components is not selected wisely.

**LDA**

LDA is a dimensionality reduction technique used to classify different classes based on the features of the supervised data. The major drawback of LDA is the so-called Small Sample Size (SSS) Problem and non-Linearity problem. SSS Problem occurs when the sample (training dataset) is quite small as compared to the dimension of the data.

**Cosine Similarity**

The measure of cosine angle between two vectors is known as the cosine similarity between two vectors, the closer the cosine value to 1 and greater will be the possibility of a match. One vector among the two vectors is the test data (detected face) and the other is the vector of the training dataset (one of the detected faces of train dataset). But it gives false results for sparse numeric data.

**HOG** 

HOG only uses magnitude values of pixels without including the neighboring values which lead to the extraction of improper features during image rotation.

**SIFT** 

SIFT is relatively alike to a sparse descriptor. Sparse Descriptor is a technique that initially detects the key points in the image and then generates descriptors at these points. SIFT consists of scale rotation and affine transformation properties as well. But it requires a long-running time as compared to other systems.

**VGG** 

VGG uses various architectures such as VGGFace1, VGGFace2 by Keras. The basic difference among these models is the number of layers included in its architecture that varies from model to model. These models have quite a good accuracy. 

**FaceRecognition API** 

FaceRecognition API is easier to use. It has a much easier architecture to implement with some inbuilt libraries required for feature recognition. You need to upload a picture and call the FaceRecognition API. The API then simulates the browser using the user’s information to call the recognition points. It works well in real-time and holds good accuracy.

**FaceNet Keras**

FaceNet Keras is a one-shot learning model. It fetches 128 vector embeddings as a feature extractor. It is even preferable in cases where we have a scarcity of datasets. It consists of good accuracy even for such situations.
This will be the model used in this project

## 3. Feature Classification
Feature classification is a geometry-based or template-based algorithm used to classify the features of the test data among different classes of facial features in the training data. These template-based classifications are possible using various statistical approaches.

The well-known methods used in feature classification can be given as:
1. Euclidean Distance
2. Cosine Similarity
3. SVM
4. KNN
5. ANN

**Euclidean Distance**

It is a distance-based feature classification method that calculates the distance between the facial nodes and the face which has the minimum difference between these distance values is considered to be the match. But it is suitable for the datasets having a smaller number of classes and lower dimensionality features.

**Cosine Similarity** 

In cosine similarity, the solution that we obtain after calculating the cosine of an angle is brought into concern. Here, we would compare the differences between these results. The more the value is closer to 1, the greater is the probability of the match. But it may give a false result if the test data features are incomplete (i.e. if the resultant value is 0 then the features don’t match, and if nearly all the features match then the value is 1).

**SVM**

SVM (Support vector machine) creates an optimal hyperplane to classify the classes of training dataset based on the different features of the face. The dimensionality of the hyperplane is one less than the number of features. Different kernels can be applied to see what features are used by the classifier to remove the features if required. This can help to improve speed.
This will be the method used in this project


**KNN** 

KNN (K-Nearest Neighbor) is all about the number of neighbors i.e. the k value. In KNN, if k=3 then we check that the data is close to which 3 data points. Thereafter, it is decided that the majority of closest data points belong to which class. Now, the test data is predicted to be in this class KNN has curse of dimensionality problem which can be solved by applying PCA before using KNN classifier.

**ANN**

ANN (Artificial Neural Network) uses a very detailed algorithm for face recognition. It classifies the local texture using multi-layer perceptron for face alignment. It uses geometric feature based and independent component analysis for feature extraction and multi artificial neural network for feature matching. This is the best method to use.
