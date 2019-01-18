function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
thetaReg = (theta(2:size(theta),:));

H = 1./(1 + exp(-((theta'*X')')));
J =  (-1 * sum((y.*log(H)) + ((1 -y).*log(1-H)))/m) + ((lambda * sum(thetaReg.^2))/(2*m));


thetaWithFirstZero = theta;
thetaWithFirstZero(1) = 0;
grad = ((X'*(H-y))/m) + lambda/m * thetaWithFirstZero;

% =============================================================

end
