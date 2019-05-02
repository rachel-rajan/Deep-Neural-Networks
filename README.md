# Malaria Detection using Deep Learning Architectures

# Introduction
Malaria is one among the deadly, infectious, and mosquito-borne disease caused by
Plasmodium parasites which are produced by the female anopheles’ mosquito. These
parasites can grow in a human body for a long run without causing symptoms and
can lead to severe complications if left unattended. Early detection of the same can
save any lives. According to WHO report the places mainly affected by malaria are
parts of Africa, South America, Asia etc.,
Several methods have been used for malarial detection and diagnosis. The paper
published by Rajaraman et al. [1], on "Pre-trained convolutional neural networks as
feature extractors toward improved Malaria parasite detection in thin blood smear
images," explains some of the methods called as polymerase chain reaction (PCR)
and rapid diagnostic tests (RDT), which are used where advanced microscopic
option are unavailable.

# Methods
Diagnosis of blood-smears is a difficult procedure to classify parasitized and
uninfected cells and is a tedious task, which needs high expertise and not reliable. In
the current period, this problem can be tackled using advanced image processing
techniques, extract relevant features and build machine learning based classification
models.
Deep learning models known as Convolution Neural Networks(CNN) are coined as
much effective learning algorithms in computer vision studies. The main layers in a
CNN is convolution and pooling layers [2]. Convolution layers can learn spatial
patterns from data, are translation invariant and are able to study different aspects of
the image such as the first layer learns about the edges and corners of the image, the
second layer learns about larger patterns obtained from first layer, and goes on.
Hence, this way feature engineering happens, and effective features are learned.
Pooling layers helps in dimensionality reduction and down sampling.
Malarial detection using this deep learning model is quite scalable and effective.
Also, with the introduction of Transfer learning this goal can be achieved much
faster even with less data constraints. As cited by [1] the deep learning model
achieved 95.9 % accuracy in detecting malaria. In this report, CNN models and pretrained
models using transfer learning approach is used to visualize the results on
the same dataset. Here MATLAB is used to build the models.

# Results and Conclusions
The data used in the analysis is from Lister Hill National Center for Biomedical
Communications (LHNCBC), which is a part of the National Library of Medicine
(NLM), it’s a publicly available dataset [3] of healthy and infected blood smear
images. The dataset consists of two folders Parasitized and Uninfected. It is a
uniform dataset with 13779 malaria and 13779 uninfected cell images.
To create a data frame, the MATLAB function called imageDatastore was used.
Since, our motive is to build deep learning models, we need training and test
dataset. In this case we divide the dataset into 80% training and 20% testing. And
then we divide the training dataset into 90% training and 10 % validation datasets.
During training, we use the training and validation dataset, and to check the
performance of the model we use Test dataset.
Here the images are resized to (125, 125, 3), and color constancy [4] is applied to it.
A color constancy is a color correction method which ensures that the color of the
image perceived by human under varying illumination conditions remains relatively
constant. Here the color correction method used is gray world assumption algorithm.
This algorithm is based on the principle that the world is gray and deduce that the
average pixel value of an uint 8-bit image is 127.5. In this way, the illumination on
different channel is eliminated independently.
Based on these images it can be seen that there are some slight differences between
the parasitized and uninfected cells. During model training, the deep learning models
try to learn patterns from these. Some of the basic configurations done are, the image
is resized to (125, 125,3), number of classes are 2, epochs are 25, and mini batch
size is 32. Three deep learning models are built in the model training phase which
are trained using the training data and performance evaluation is done using the
validation data. In our CNN model architecture, we have 3 convolution and pooling
layers, two dense layers, and dropout regularization, then the model is trained. The
validation accuracy obtained is 94.10%, which is a good result, usage of L2
regularization reduces overfitting.

# Transfer Learning
Transfer learning enables us to utilize the resources from previously learned tasks
and use in deep learning models. Here a pre-trained deep learning model is used to
detect malarial parasites. They can be used for fine tuning or as a feature extractor.
Here, we use the pre-trained AlexNet deep learning model, developed by Alex
Krizhevsky [2], and published with Ilya Sutskever and Krizhevsky's PhD advisor
Geoffrey Hinton. This model was trained on a huge dataset called ImageNet. They
won the 2012 ImageNet LSVRC-2012 competition model.
The highlights of the network they used ReLu to add non-linearity which speeds up
6 times with the same accuracy. Dropout was used instead of Regularization to
reduce overfitting, but the training time is doubled. Overlap pooling was used to
reduce the size of the network. The architecture of AlexNet consists of 5
convolutional layers and 5 fully connected layers. After every convolutional and
fully connected layers ReLu is applied. Before the first and second fully connected
layer Dropout is applied. The image size is (227,227,3) in the following architecture.

In our analysis with the help of Deep Network Designer in MATLAB, we can alter
the layers in AlexNet according to our analysis. Here in the last fully connected
layer, the output size is changed to the number of classes which is 2, and the weight
learn, and bias learn rate factor are changes to 20 to improve the performance, and
then a new classification layer is added. Learning curves for AlexNet CNN
architecture is shown in Fig 7. After training, it gives us a validation accuracy of
almost 95.96 % and, based on the training accuracy, it can be observed that it’s not
overfitting.
Performance Evaluation
We have the customized CNN model with color constancy and pre-trained AlexNet
model. Now we test the performance of the model on the actual test dataset. With
the saved deep learning models, and scaling the test data, we make predictions on
the test dataset. From the results it can be seen that the AlexNet model performs
better compared to the customized CNN model with color consistency.

# Reference
[1] Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude, RJ,
Jaeger S, Thoma GR. (2018) Pre-trained convolutional neural networks as
feature extractors toward improved Malaria parasite detection in thin blood
smear images. PeerJ6:e4568 https://doi.org/10.7717/peerj.4568 \
[2] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2017. ImageNet
classification with deep convolutional neural networks. Commun. ACM 60,
6 (May 2017), 84-90. DOI: https://doi.org/10.1145/3065386 \
[3] Poostchi et al. (2018) Poostchi M, Silamut K, Maude RJ, Jaeger S, Thoma
GR. Image analysis and machine learning for detecting malaria.
Translational Research. 2018;194:36–55. doi: 10.1016/j.trsl.2017.12.004\
[4] Cepeda-Negrete, Jonathan & Sanchez-Yanez, Raul. (2011). Color Constancy
Algorithms in Practice. 10.13140/RG.2.1.1956.6569 <br\>

