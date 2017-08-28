%This code was used to create data sets
%in CSV form for the neural nets to train with.
%The images were a subset taken from the Essex set, cleaned 
%up, seperated into Male and Female examples,
%and fed to this Matlab script, in color and grayscale versions 
%Copyright David Kaplan 2016-2017


num_female_training = 341;
num_male_training = 342;

num_female_test = 38;
num_male_test = 38;

nrow = 200;
ncol = 180;

noutputs = 2;
male_port_0 = 0.05;
male_port_1 = 0.95;
female_port_0 = 0.95;
female_port_1 = 0.05;

% Stack the images, male and female, in flattened row vector vectors into a 2d array
% The first num_female_training row or num_female_test rows are female
% followed by num_male_training rows or num_male test rows
%
training_data_gray = zeros(num_female_training + num_male_training, nrow*ncol + noutputs);
training_data_sxs = zeros(num_female_training + num_male_training, nrow*ncol*3 + noutputs);

test_data_gray = zeros( num_female_test + num_male_test, nrow*ncol + noutputs);
test_data_sxs = zeros(num_female_test + num_male_test, nrow*ncol*3 + noutputs);

divisor = 255.0*ones(1,nrow*ncol);
divisor2 = 255.0*ones(1,nrow*ncol*3);

 % load the training examples
 for i = 1:(num_female_training+num_male_training)
     filename = cat(2,num2str(i), '_imgray.bmp');
     filename = cat(2,'all_training/', filename);
     im =  imread(filename);
     data = double((reshape(im,1, nrow*ncol)))./divisor;
     training_data_gray(i,1:nrow*ncol) = data;
     
     %add the output ports for the neural net
     if (i <= num_female_training)
         % female example
         training_data_gray(i,nrow*ncol + 1) = female_port_0;
         training_data_gray(i,nrow*ncol + 2) = female_port_1;
     else
         training_data_gray(i,nrow*ncol + 1) = male_port_0;
         training_data_gray(i,nrow*ncol + 2) = male_port_1;
     end
     
 end
 
  % load the test examples
 for i = 1:(num_female_test+num_male_test)
     filename = cat(2,num2str(i), '_imgray.bmp');
     filename = cat(2,'all_test/', filename);
     im =  imread(filename);
     data = double((reshape(im,1, nrow*ncol)))./divisor;
     test_data_gray(i,1:nrow*ncol) = data;
      %add the output ports for the neural net
     if (i <= num_female_test)
         % female example
         test_data_gray(i,nrow*ncol + 1) = female_port_0;
         test_data_gray(i,nrow*ncol + 2) = female_port_1;
     else
         test_data_gray(i,nrow*ncol + 1) = male_port_0;
         test_data_gray(i,nrow*ncol + 2) = male_port_1;
     end
 end
 
 clear im;
%  %this will display image 1: imagesc(reshape(training_data(1,:), nrow, ncol))


 
 
%this section creates the 3 stacked side by side images, flattened 
%to row vectors, all in one array, for the training data
for i = 1:(num_female_training+num_male_training)
    filename = cat(2, num2str(i), '.jpg');
    filename = cat(2, 'all_training/', filename);
    im =  imread(filename);
    
    R = im(:,:,1); 
    G = im(:,:,2);
    B = im(:,:,3);


    sxs = zeros(nrow, 3*ncol);
    sxs(:,1:ncol) = R;
    sxs(:,ncol+1:2*ncol) = G;
    sxs(:,2*ncol+1:3*ncol) = B;
    
    data2 = double((reshape(sxs,1, nrow*ncol*3)))./divisor2;
    training_data_sxs(i,1:nrow*ncol*3) = data2;
    if (i <= num_female_training)
         % female example
         training_data_sxs(i,nrow*ncol*3 + 1) = female_port_0;
         training_data_sxs(i,nrow*ncol*3 + 2) = female_port_1;
     else
         training_data_sxs(i,nrow*ncol*3 + 1) = male_port_0;
         training_data_sxs(i,nrow*ncol*3 + 2) = male_port_1;
     end

end
clear R G B sxs;

%this section creates the 3 stacked side by side images, flattened 
%to row vectors, all in one array, for the test data
for i = 1:(num_female_test+num_male_test)
    filename = cat(2, num2str(i), '.jpg');
    filename = cat(2, 'all_test/', filename);
    im =  imread(filename);
    
    R = im(:,:,1); 
    G = im(:,:,2);
    B = im(:,:,3);


    sxs = zeros(nrow, 3*ncol);
    sxs(:,1:ncol) = R;
    sxs(:,ncol+1:2*ncol) = G;
    sxs(:,2*ncol+1:3*ncol) = B;
    
    data2 = double((reshape(sxs,1, nrow*ncol*3)))./divisor2;
    test_data_sxs(i,1:nrow*ncol*3) = data2;
    if (i <= num_female_test)
         % female example
         test_data_sxs(i,nrow*ncol*3 + 1) = female_port_0;
         test_data_sxs(i,nrow*ncol*3 + 2) = female_port_1;
     else
         test_data_sxs(i,nrow*ncol*3 + 1) = male_port_0;
         test_data_sxs(i,nrow*ncol*3 + 2) = male_port_1;
     end
end
clear R G B sxs;




% scratch
% w = Ttest_gray_female(1,:);
% fproj_w = w*female_subset_eigs_gray';
% imagesc(reshape(fproj_w+avg_female_gray, nrow, ncol));
