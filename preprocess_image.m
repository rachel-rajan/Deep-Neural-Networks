function [Iout] = preprocess_image(filename)
%UNTITLED Summary of this function preprocess the road sign image using
%reshape
I = imread(filename);
% I = rgb2gray(I);
I = ColorConstancy(I);
Iout = imresize3(I, [125,125,3]); % downsampling compared to upsampling

end