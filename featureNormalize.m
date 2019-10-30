function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Inintialization
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% store the means and standard deviations for the features
for i = 1 : size(X, 2) 
    mu(i) = mean(X(:, i));
    sigma(i) = std(X(:, i));
end

% subtract the mean from the features and divide the features by std to
% complete the normalization
for i = 1 : size(X, 2)
    X_norm(:, i) = X(:, i) - mu(i);
    X_norm(:, i) = X_norm(:, i) / sigma(i);
end
end
