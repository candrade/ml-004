function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
			
		% Remember: h_theta(x) = (theta_0 * x_0 + theta_1 * x_1); where x_0 = 1
		% or [x_0 x_1] * [theta_0; theta_1] and results in a vector.
		% How do I represent x^(i) or the ith training example?
		% X * theta => will always produce a vector the size of m_x_1
		% The training examples will always m_x_n so in order to properly
		% vectorize we must transpose (hypothesis - y) and multiply by X
		% There was an error in trying in my first calculation in that I did not need a sum.
		% I was trying to perform the following
		% delta = (1 / m) * sum((X * theta - y)' * X);
		% The sum is implied by the matrix math already.
		delta = (1 / m) * (X' * (X * theta - y));
		theta = theta - (alpha * delta);

    % ============================================================
		%fprintf('%f %f \n', theta(1), theta(2));
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end