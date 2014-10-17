function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vals = [0.01 0.03 0.1 0.3 1 3 10]; 
sigma_vals = [0.01 0.03 0.1 0.3 1 3 10];
err = zeros(length(C_vals), length(sigma_vals));

for i = 1:length(C_vals)
  for j = 1:length(sigma_vals)
		% determine model based on C(i) using svmTrain.m
		model = svmTrain(X, y, C_vals(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vals(j)));
		% determine prediction using model as input to svmPredict.m
		predictions = svmPredict(model, Xval);
		% calculate error using the method suggested in the assignment notes: mean(double(predictions ~= yval)) 
		% NOTE: store your erro in a matrix at location i,j
		err(i,j) = mean(double(predictions ~= yval));
  end
end

[minval,ind] = min(err(:));
[I,J] = ind2sub([size(err,1) size(err,2)],ind);

C = C_vals(I);
sigma = sigma_vals(J);
% =========================================================================

end
