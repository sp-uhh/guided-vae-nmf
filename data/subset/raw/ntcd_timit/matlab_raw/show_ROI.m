%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                              
% Written by Ahmed Hussen Abdelaziz 
% Copyright(c) ICSI 2016                                  
% Permission is granted for anyone to copy, use, or modify 
% this program for purposes of research or education. This program 
% is distributed without any warranty express or implied. 
%
% If you use this script please cite:
% A. H. Abdelaziz, “NTCD-TIMIT: A New Database and Baseline for 
% Noise-robust Audio-visual Speech Recognition,” In Proc. Interspeech, 
% 2017.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script is used to show gray-scaled images from 
% the DCT visual features 
%
clc
clear
close all
warning off

% Data set
data_set     = 'train'; %train, dev, test

speakers     = dir(data_set);
speakers     = speakers(3:end);

% Choose Speaker
choose_spk   = 5; % <length(speakers)

% Dimension of the normalized mouth
width = 67;
hight = 67;

% Length after DCT
d     = width * hight;
    
read_dir  = [ data_set '\' speakers(choose_spk).name '\'];

file_list = dir([read_dir '*.mat']);

n_files   = length(file_list);

% Choose an utterance
choose_utt   = 1; % < n_files

load([read_dir file_list(choose_utt).name])

% Number of frames in the chosen utterance
n_frames = size(data,2);

% Initialize the mouth matrix that contains all images of this utterance
mouth     = zeros(hight,width,1,n_frames);

for i = 1:n_frames
    mouth(:,:,1,i) = idct2(reshape(data(:,i),[hight,width]));
end

% Choose frame to be displayed
choose_frame   = 20; % < n_frames

figure
% Show the chosen frame
imshow(mouth(:,:,1,i),[0,255])


