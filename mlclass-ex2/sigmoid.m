function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Remember: 
% 1 / (1 + e ^ -z); where z = theta' * x
% The question to answer is how do we apply the calc
% to every value of z?
% For example you define z=[4,-5,2,0,4,5]. Output will be [g(4),g(-5)...]

% exp(x) => Compute e^x for each element of x
g = 1 ./ (1 .+ exp(-z));


% =============================================================

end
