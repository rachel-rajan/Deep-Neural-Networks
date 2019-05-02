%% Malaria detection using AlexNet

%clear all
clc; clear all; close all;

categories = {'Parasitized', 'Uninfected'};
folder = 'C:\Users\Rachel Rajan\Desktop\Books\Books\ML_DL\Project2\cell_images\';

% Read data
imds = imageDatastore(fullfile(folder, categories), 'LabelSource', 'foldernames');

%%
imds.ReadFcn = @(filename)preprocess_image(filename);

train_percentage = 0.8;
[imdsTrain, imdsTest] = splitEachLabel(imds,train_percentage);

train_percentage = 0.9;
[imdsTrain, imdsValid] = splitEachLabel(imdsTrain,train_percentage);

%% AlexNet deep learning model

net = alexnet;

deepNetworkDesigner

%Display the network architecture.
net.Layers

%%
inputSize = net.Layers(1).InputSize;

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValid);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%%
miniBatchSize = 32;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% train network

trainedNet = trainNetwork(augimdsTrain,layers_1,options);

%%
%load net01

layer = 'fc7';
featuresTrain = activations(trainedNet,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(et,augimdsTest,layer,'OutputAs','rows');

%Extract the class labels from the training and test data.

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%% classify test images
classifier = fitcecoc(featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);
Display four sample test images with their predicted labels.

idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

plotconfusion(YTest,YPred);