function [jVal, grad] = costFunction(theta)

jVal = (theta(1, 1) - 5)^2 + (theta(2, 1) - 5)^2;

grad = zeros(size(theta, 1), 1);
grad(1, 1) = 2 * (theta(1, 1) - 5);
grad(2, 1) = 2 * (theta(2, 1) - 5);

end