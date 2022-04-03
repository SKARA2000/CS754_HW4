%% Clearing console and variables
clc; clear all;
%% Reading the image and adding noise to it
imagePath = "../barbara256.png";
X = double(imread(imagePath));
paddedX = padarray(X, [4, 4], 'replicate', 'both');
% paddedX = padarray(X, [4, 4], 0, 'both');
tic;
%% ISTA algorithm
U = kron(dctmtx(8)', dctmtx(8)');           % We are using 8x8 patches to reconstruct x
rng(80);
phi = randn(32, 64);                        % measurement matrix is a random gaussian matrix of size 32x64 here
A = phi*U;                              
alpha = max(eig(A'*A)) + 1;                 % alpha is the parameter we will be using in the ISTA algo
[rows, cols] = size(X);
reconstructed_img = zeros(size(paddedX));
counts = zeros(size(paddedX));
for i=1:rows
    for j=1:cols
        patch = phi*reshape(paddedX(i:i+7, j:j+7), 64,1);                                           % We are considering an 8x8 patch over here
        lambda = 1;
        theta = zeros([64,1]);                                                                      % Initial DCT coefficients are random for the algo
        for k=1:300
            theta = softhresh((theta + A'*(patch - A*theta)*(1/alpha)), (lambda/(2*alpha)));        % Soft thresholding with a pre-determined number of 
                                                                                                    % iterations to calculate the DCT coefficients for each patch
        end
        reconstructed_patch = reshape(U*theta, 8, 8);
        reconstructed_img(i:i+7, j:j+7) = reconstructed_img(i:i+7, j:j+7) + reconstructed_patch;
        counts(i:i+7, j:j+7) = counts(i:i+7, j:j+7) + ones(8);                                      % Update the counts matrix to help in averaging
    end 
end
final_image = reconstructed_img./counts;            % Average is calcualated by simply dividing the reconstructed image with
                                                    % the counts matrix
X_hat = final_image(5:rows+4, 5:cols+4);
X_hat = X_hat*2;
rmse = norm(X-X_hat, 'fro')/norm(X, 'fro');
%% Printing plots
fprintf("RMSE error = %.5f\n", rmse);
figure();
subplot(1,2,1);
imshow(uint8(X));
title('Original Image');
subplot(1,2,2);
imshow(uint8(X_hat));
title('Reconstructed Image');
saveas(gcf(), '../images/part1b/recon.png')
toc;