%% Clearing console and variables
clc; clear all;
%% Reading the image and adding noise to it
imagePath = "../barbara256.png";
X = imread(imagePath);
paddedX = padarray(X, [4, 4], 'replicate', 'both'); % Padding the image for easy avergaing of the kernels in the later stages
% paddedX = padarray(X, [4, 4], 0, 'both');
rng(42);
variance = 3;                                       
N = sqrt(variance).*randn(size(paddedX));           % Noise vector to be added to the image as per spec
Y = cast(double(paddedX) + N, 'uint8');             % Noisy image

% Y1 = imnoise(X, 'gaussian', 0, 3);
figure();
subplot(1,2,2);
imshow(Y);
title('Noisy Image');
subplot(1,2,1)
imshow(X);
title('Original Image');
saveas(gcf(), '../images/part1a/noisy.png');
tic;
%% ISTA algorithm
U = kron(dctmtx(8)', dctmtx(8)');           % We are using 8x8 patches to reconstruct x
phi = eye(64);                              % measurement matrix is the general sampling matrix in this case
A = phi*U;                          
alpha = max(eig(A'*A)) + 1;                 % alpha is the parameter we will be using in the ISTA algo
[rows, cols] = size(X);
reconstructed_img = zeros(size(paddedX));
counts = zeros(size(paddedX));
for i=1:rows
    for j=1:cols
        patch = double(reshape(Y(i:i+7, j:j+7), 64,1));                                         % We are considering an 8x8 patch over here
        lambda = 1;
        theta = randn([64,1]);                                                                  % Initial DCT coefficients are random for the algo
        for k=1:100
            theta = softhresh((theta + A'*(patch - A*theta)*(1/alpha)), (lambda/(2*alpha)));    % Soft thresholding with a pre-determined number of 
                                                                                                % iterations to calculate the DCT coefficients for each patch
        end
        reconstructed_patch = reshape(U*theta, 8, 8);
        reconstructed_img(i:i+7, j:j+7) = reconstructed_img(i:i+7, j:j+7) + reconstructed_patch;
        counts(i:i+7, j:j+7) = counts(i:i+7, j:j+7) + ones(8);                                  % Update the counts matrix to help in averaging
    end 
end
reconstructed_img = reconstructed_img./counts;      % Average is calcualated by simply dividing the reconstructed image with
                                                    % the counts matrix
X_hat = reconstructed_img(5:rows+4, 5:cols+4);
rmse = norm(double(X)-X_hat, 'fro')/norm(double(X), 'fro');
%% Printing plots
fprintf("RMSE error = %.5f\n", rmse);
figure();
subplot(1,3,1);
imshow(X);
title('Original Image');
subplot(1,3,2);
imshow(Y);
title('Noisy Image');
subplot(1,3,3);
imshow(uint8(X_hat));
title('Reconstructed Image');
saveas(gcf(), '../images/part1a/recon.png');
toc;