function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% a_1 = bias(X);
% z_2 = a_1 * Theta1';
% a_2 = sigmoid(z_2);
% a_2 = bias(a_2);
% z_3 = a_2 * Theta2';
% a_3 = sigmoid(z_3);

yv = repmat(1:num_labels, size(y,1) , 1) == repmat(y, 1, num_labels);

for i=1:m	
  for k=1:num_labels		
    J += (-yv(i,k) * log(a_3(i,k))) - ((1 - yv(i,k)) * log( 1 - a_3(i,k)));
  end
end

t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);
l = lambda / (2 * m) * (sum(sum(t1 .^ 2)) + sum(sum(t2 .^ 2)));

J = 1/m * J + l;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% Non-Regularized Unit Test
% [J grad] = nnCostFunction([1 2 3 2 4 1 2 1 3 4 3 2 3 2 3 5 2 1 4 1 2 3 5 4],3,3, 3, [1 2 3;4 6 5;3 2 3], [3; 2; 3], 0)

% J =  25.000
% grad =
% 
%    1.3312e-08
%    6.9020e-08
%    3.4069e-07
%    1.3791e-08
%    6.9066e-08
%    4.2191e-07
%    2.6625e-08
%    1.3804e-07
%    6.8138e-07
%    3.9937e-08
%    2.0706e-07

% Regularized Unit Test
% [J grad] = nnCostFunction([1 2 3 2 4 1 2 1 3 4 3 2 3 2 3 5 2 1 4 1 2 3 5 4],3,3,3,[1 2 3;4 6 5;3 2 3],[3; 2; 3], 1)
% 
% J =  52.500
% grad =
% 
%    1.3312e-08
%    6.9020e-08
%    3.4069e-07
%    6.6667e-01
%    1.3333e+00
%    3.3333e-01
%    6.6667e-01
%    3.3333e-01
%    1.0000e+00
%    1.3333e+00

% I used the following steps:

a_1 = bias(X);
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);
a_2 = bias(a_2);
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

%fprintf(['a3 size: (%f, %f) \n'], size(a_3));

% Compute delta3 by subtracting yv from a3 - matrix[5000x10]
delta_3 = a_3 - yv;

% Compute intermediate r2 by multiplying delta3 and Theta2 (without first column of θ(2)0 elements) - matrix[5000x25]
% Compute delta2 by per-element multiplication of z2 passed through sigmoidGradient and r2 - matrix[5000x25]
delta_2 = (delta_3 * Theta2(:,2:end)) .* sigmoidGradient(z_2);

% Compute regularization term t1 by multiplying Theta1 by lambda scalar and then setting first column to zero to account for j=0 case - matrix[25x401]
t1 = lambda/m * Theta1;
t1(:,1) = 0;

% Compute Theta1_grad (D(2)) by multiplying transposed delta2 and X, adding the regularization term t1 and then dividing by scalar m - matrix[25x401]
Theta1_grad = 1/m * (delta_2' * a_1) + t1;

% Compute second regularization term t2 in the same manner, but for Theta2 - matrix[10x26]
t2 = lambda/m * Theta2;
t2(:,1) = 0;

% Compute Theta2_grad (D(L)) in the same manner as Theta1_grad - matrix[10x26]
Theta2_grad = 1/m * (delta_3' * a_2) + t2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
