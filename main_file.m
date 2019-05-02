%Clear all
clc; clear all; close all;

categories = {'Parasitized', 'Uninfected'};
folder = 'C:\Users\Rachel Rajan\Desktop\Books\Books\ML_DL\Project2\cell_images\';
% Read data
imds = imageDatastore(fullfile(folder, categories), 'LabelSource', 'foldernames');

%Dataset count
total_count = countEachLabel(imds);
total_count.Count;

figure;
bar(total_count.Count)
title('Dataset Distribution');
xlabel('class #')
ylabel('count')

%Convert Categorical to Double
labels = double(imds.Labels);

%For loop
num_classes = length(unique(labels));
for i = 1:num_classes
%Find command
cat_idx = find(labels == i,1,'first');

%Read the image
I = readimage(imds, cat_idx);
figure;
imagesc(I);
title(sprintf('Class: %0d', i))
size(I);
pause;
end

%%
imds.ReadFcn = @(filename)preprocess_image(filename);

train_percentage = 0.8;
[imdsTrain, imdsTest] = splitEachLabel(imds,train_percentage,'randomized');

train_percentage = 0.9;
[imdsTrain, imdsValid] = splitEachLabel(imdsTrain,train_percentage,'randomized');

%count number of labels

train_count = countEachLabel(imdsTrain);
valid_count = countEachLabel(imdsValid);
bar([train_count.Count, valid_count.Count])
xlabel('class #')
ylabel('count')

%% architecture of the custom model
layers = [ ...
imageInputLayer([125 125 3],'Normalization', 'zerocenter')

convolution2dLayer(3,32,'Stride',1)
%batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,64,'Stride',1)
%batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,128,'Stride',1)
%batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,256,'Stride',1)
%batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,512,'Stride',1)
%batchNormalizationLayer
reluLayer
dropoutLayer
fullyConnectedLayer(512)
reluLayer
dropoutLayer
fullyConnectedLayer(2)
softmaxLayer 
classificationLayer()];

% declare training options
options = trainingOptions('sgdm','Verbose',true,'LearnRateSchedule','none','L2Regularization',3.6e-06,'MaxEpochs',25,'MiniBatchSize',32,...
      'ValidationData',{imdsValid,imdsValid.Labels},'ValidationFrequency',30,'ValidationPatience',10,'Plots','training-progress',...
        'Momentum',0.99, 'InitialLearnRate',1.9935e-04); 
covnet = trainNetwork(imdsTrain, layers, options); 

%% Feature Visualization

covnet.Layers 

%% 

layer = 'fc_2';
featuresTrain = activations(covnet,imdsTrain,layer,'OutputAs','rows');
featuresTest = activations(covnet,imdsTest,layer,'OutputAs','rows');

% Extract the class labels from the training and test data.

YTrain = imdsTrain.Labels;
classifier = fitcecoc(featuresTrain,YTrain);

%% Classify the test images using the trained SVM model the features extracted from the test images.

YPred = predict(classifier,featuresTest);
YTest = imdsTest.Labels;

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