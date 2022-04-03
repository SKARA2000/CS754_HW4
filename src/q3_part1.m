%% Clearing concole and variables
clc; clear all;
addpath('../l1_ls_matlab');
%% Reading and padding the brain slice 
num_slices = 6;                                                                 % Number of brain slices processed
img_path = '../slice_%d.png';
for i=1:num_slices
    slices(:,:,i) = im2double(imread(sprintf(img_path, 49+i)));
end
[rows, cols] = size(slices(:,:,1));
for i=1:num_slices
    % We put a 0 padding to make an image of size 255x255. Other options
    % are commented out but can be checked for further RMSE reductions
    padded_slices(:,:,i) = padarray(slices(:,:,i), [(255-rows)/2, (255-cols)/2], 0, 'both');    
%     padded_slices(:,:,i) = padarray(slices(:,:,i), [(255-rows)/2, (255-cols)/2], 'replicate', 'both');
%     padded_slices(:,:,i) = padarray(slices(:,:,i), [(cols-rows)/2, 0], 0, 'both');
%     padded_slices(:,:,i) = padarray(slices(:,:,i), [(cols-rows)/2, 0], 'replicate', 'both');
end
%Comparison of original witht he padded image
figure();
subplot(1,2,1);
imshow(slices(:,:,i));
title('Original slice(slice-50)');
axis on; axis tight; colormap('gray'); colorbar;
subplot(1,2,2);
imshow(padded_slices(:,:,i));
title('Padded slice(slice-50)');
axis on; axis tight; colormap('gray'); colorbar;
saveas(gcf(), '../images/padded.png');
%% Filtered back-projection using Ram-Lak filter
% By default, the angles are taken to be uniformly spaced. Random angles
% can also be chopsen for the same.
angles = 0:10:179;
% angles = unifrnd(0, 180, 1, 18);
for i=1:num_slices
    [Y, xp] = radon(padded_slices(:,:,i), angles);                                  % Simple radon projections for each of the 18 angles
    original_size = size(padded_slices(:,:,i), 1);
    figure();
    imshow(Y,[],'Xdata',angles,'Ydata',xp,'InitialMagnification','fit');
    xlabel('\theta {degrees}');
    ylabel('x''');
    colormap(gca, hot), colorbar;
    title(sprintf('Sinogram with the limited radon projections of slice-%d', 49+i));

    reconstructed_image = iradon(Y, angles, 'linear', 'Ram-Lak', 1, original_size); % Reconstructed image using inverse radon-transformation
%     figure();
%     subplot(1, 2, 1);
%     imshow(padded_slices(:,:,i));
%     title(sprintf('Original Padded slice(slice-%d)', 49+i));
%     axis on; axis tight; colormap('gray'); colorbar;
%     subplot(1, 2, 2);
%     imshow(reconstructed_image); 
%     title(sprintf('Reconstructed Padded slice(slice-%d)', 49+i));
%     axis on; axis tight; colormap('gray'); colorbar;
    figure();
    imshow(reconstructed_image); 
    title(sprintf('Reconstructed Padded slice(slice-%d)', 49+i));
    axis on; axis tight; colormap('gray'); colorbar;
%     saveas(gcf(), sprintf('../images/random/ram_lak_%d.png', 49+i));
    saveas(gcf(), sprintf('../images/uniform/ram_lak_%d.png', 49+i));
end
%% Independent CS-based tomographic reconstruction 
tic;
for i=1:num_slices
    slice = padded_slices(:,:,i);
    Y = radon(slice, angles);
    measurement_size = size(Y, 1);
    original_size = size(slice, 1);
    m = size(Y(:), 1);
    n = size(slice(:), 1);  
    A = forward_handler(@idct2, @radon, measurement_size, original_size, angles);       % Class object that overloads the matrix multiplication operator to 
                                                                                        % to use radon and 2d-idct transform handles to compute the forward 
                                                                                        % model of the radon transformation of a vector
    At = forward_handler_t(@dct2, @iradon, measurement_size, original_size, angles);    % Same logic but for computing A'
    lambda = 0.1;
    rel_tol = 1e-9;
    [Beta, status] = l1_ls(A, At, m, n, Y(:), lambda, ...                               
        rel_tol);                                       % Using the class-based l1_ls solver for computing the approximated solution 
                                                        % of the l1-regularised least squares problem
    if strcmpi(status, 'solved')
        disp("Linear Problem Solved");
    else
        disp("Linear Problem unsolved");
    end
    reconstructed_image = idct2(reshape(Beta, original_size, original_size));   % Reconstructing the original image using the 2D-inverse DCT of the solved vector
%     figure();
%     subplot(1, 2, 1);
%     imshow(slice);
%     title(sprintf("Original slice(slice-%d)", 49+i));
%     axis on; axis tight; colormap('gray'); colorbar;
%     subplot(1, 2, 2);
%     imshow(reconstructed_image);
%     title(sprintf("CS-based reconstructed slice-%d", 49+i));
%     axis on; axis tight; colormap('gray'); colorbar;
    figure();
    imshow(reconstructed_image);
    title(sprintf("CS-based reconstructed slice-%d", 49+i));
    axis on; axis tight; colormap('gray'); colorbar;    
%     saveas(gcf(), sprintf('../images/random/ind_CS_%d.png', 49+i));
    saveas(gcf(), sprintf('../images/uniform/ind_CS_%d.png', 49+i));
end
toc;
%% Coupled CS-based tomographic reconstruction
tic;
% Angle sets considered for Coupled CS-based recosntruction is uniformly
% spaced by default but uses different angle sets. 
angles_1 = 0:10:179;        
angles_2 = 5:10:179;
% angles_1 = unifrnd(0, 180, 1, 18);
% angles_2 = unifrnd(0, 180, 1, 18);
for i=1:num_slices/2
    slice_1 = padded_slices(:,:,2*i-1);
    slice_2 = padded_slices(:,:,2*i);
    Y_1 = radon(slice_1 , angles_1);
    Y_2 = radon(slice_2 , angles_2);
    measurement_size = size(Y_1, 1);
    Y_1 = Y_1(:);
    Y_2 = Y_2(:);
    Y = [Y_1; Y_2];
    m = size(Y, 1);
    n = size(slice_1(:), 1) + size(slice_2(:), 1);
    original_size = size(slice_1, 1);
    % Class objects defined for handling matrix multiplication using
    % transform handles like in the previous section
    A = coupled_forward_handler(@idct2, @radon, measurement_size, original_size, angles_1, angles_2);
    At = coupled_forward_handler_t(@dct2, @iradon, measurement_size, original_size, angles_1, angles_2);
    lambda = 0.1;
    rel_tol = 1e-9;
    [Beta, status] = l1_ls(A, At, m, n, Y, lambda, rel_tol);
    if strcmpi(status, 'solved')
        disp("Linear Problem Solved");
    else
        disp("Linear Problem unsolved");
    end
    Beta_1 = Beta(1:0.5*n);
    delta_Beta_1 = Beta(0.5*n + 1:end);
    % Reconstruction using the inverse 2D-DCT of the solved vectors
    reconstructed_slice_1 = idct2(reshape(Beta_1, original_size, original_size));
    % Beta2 = (Beta1 + delta_Beta) by definition of the objective function
    reconstructed_slice_2 = idct2(reshape(Beta_1 + delta_Beta_1, original_size, original_size));
%     figure();
%     subplot(1, 2, 1);
%     imshow(reconstructed_slice_1);
%     title(sprintf("Coupled CS recon. slice-%d", 48 + 2*i));
%     axis on; axis tight; colormap('gray'); colorbar;
%     subplot(1, 2, 2);
%     imshow(reconstructed_slice_2);
%     title(sprintf("Coupled CS recon. slice-%d", 49 + 2*i));
%     axis on; axis tight; colormap('gray'); colorbar;
    figure();
    imshow(reconstructed_slice_1);
    title(sprintf("Coupled CS recon. slice-%d", 48 + 2*i));
    axis on; axis tight; colormap('gray'); colorbar;    
%     saveas(gcf(), sprintf('../images/random/coupled_CS_%d.png', 48+2*i));
    saveas(gcf(), sprintf('../images/uniform/coupled_CS_%d.png', 48+2*i));    
    figure();
    imshow(reconstructed_slice_2);
    title(sprintf("Coupled CS recon. slice-%d", 49 + 2*i));
    axis on; axis tight; colormap('gray'); colorbar;        
%     saveas(gcf(), sprintf('../images/random/coupled_CS_%d.png', 49+2*i));
    saveas(gcf(), sprintf('../images/uniform/coupled_CS_%d.png', 49+2*i));
end
toc;
%% Coupled CS-based reconstruction with three slices
tic;
% Angle sets considered for 3-slice Coupled CS-based recosntruction is 
% uniformly spaced by default but uses different angle sets. 
angles_1 = 0:10:179;
angles_2 = 3:10:179;
angles_3 = 6:10:179;
% angles_1 = unifrnd(0, 180, 1, 18);
% angles_2 = unifrnd(0, 180, 1, 18);
% angles_3 = unifrnd(0, 180, 1, 18);
for i=1:num_slices/3
    slice_1 = padded_slices(:,:,3*i-2);
    slice_2 = padded_slices(:,:,3*i-1);
    slice_3 = padded_slices(:,:,3*i);
    Y_1 = radon(slice_1, angles_1);
    Y_2 = radon(slice_2, angles_2);
    Y_3 = radon(slice_3, angles_3);
    Y = [Y_1(:); Y_2(:); Y_3(:)];
    m = size(Y, 1);
    n = size(slice_1(:), 1) + size(slice_2(:), 1) + size(slice_3(:), 1);
    measurement_size = size(Y_1, 1);
    original_size = size(slice_1, 1);
    % Class objects defined for handling matrix multiplication using
    % transform handles like in the previous section
    A = coupled_forward_handler_3(@idct2, @radon, measurement_size, original_size, angles_1, ...
        angles_2, angles_3);
    At = coupled_forward_handler_3t(@dct2, @iradon, measurement_size, original_size, angles_1, ...
        angles_2, angles_3);
    lambda = 0.1;
    rel_tol = 1e-9;
    disp(size(Y));
    [beta, status] = l1_ls(A, At, m, n, Y, lambda, rel_tol);
    Beta1 = beta(1:n/3);
    delta_Beta1 = beta(n/3 + 1:2*n/3);
    delta_Beta2 = beta(2*n/3 + 1:end);
    Beta2 = Beta1 + delta_Beta1;
    Beta3 = Beta1 + delta_Beta1 + delta_Beta2;
    % Reconstruction using the inverse 2D-DCT of the solved vectors
    reconstrucetd_slice_1 = idct2(reshape(Beta1, original_size, original_size));
    % Beta2 = (Beta1 + delta_Beta1) by definition of the objective function
    reconstrucetd_slice_2 = idct2(reshape(Beta2, original_size, original_size));
    % Beta2 = (Beta1 + delta_Beta2) by definition of the objective function
    reconstrucetd_slice_3 = idct2(reshape(Beta3, original_size, original_size));
%     figure();
%     subplot(3, 1, 1);
%     imshow(reconstrucetd_slice_1);
%     title(sprintf("3-slice CS recon.: slice %d", 47+3*i));
%     axis on; axis tight; colormap('gray'); colorbar;
%     subplot(3, 1, 2);
%     imshow(reconstrucetd_slice_2);
%     title(sprintf("3-slice CS recon.: slice %d", 48+3*i));
%     axis on; axis tight; colormap('gray'); colorbar;
%     subplot(3, 1, 3);
%     imshow(reconstrucetd_slice_3);
%     title(sprintf("3-slice CS recon.: slice %d", 49+3*i));
%     axis on; axis tight; colormap('gray'); colorbar;
    figure();
    imshow(reconstrucetd_slice_1);
    title(sprintf("3-slice CS recon.: slice %d", 47+3*i));
    axis on; axis tight; colormap('gray'); colorbar;    
%     saveas(gcf(), sprintf('../images/random/3coupled_CS_%d.png', 47+3*i));    
    saveas(gcf(), sprintf('../images/uniform/3coupled_CS_%d.png', 47+3*i));        
    figure();
    imshow(reconstrucetd_slice_2);
    title(sprintf("3-slice CS recon.: slice %d", 48+3*i));
    axis on; axis tight; colormap('gray'); colorbar;
%     saveas(gcf(), sprintf('../images/random/3coupled_CS_%d.png', 48+3*i));    
    saveas(gcf(), sprintf('../images/uniform/3coupled_CS_%d.png', 48+3*i));    
    figure();
    imshow(reconstrucetd_slice_3);
    title(sprintf("3-slice CS recon.: slice %d", 49+3*i));
    axis on; axis tight; colormap('gray'); colorbar;        
%     saveas(gcf(), sprintf('../images/random/3coupled_CS_%d.png', 49+3*i));
    saveas(gcf(), sprintf('../images/uniform/3coupled_CS_%d.png', 49+3*i));
end
toc;