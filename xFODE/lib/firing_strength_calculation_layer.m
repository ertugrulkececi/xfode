% 22 JUN 2023
function output = firing_strength_calculation_layer(membership_values)
% v0.2 compatibel with mini-batch
% more operators can be added
%
% rule inferance for static rules, rule count is equal to number of input membership
% function or number of outputs
%
% @param output -> output
%
%       (n,1,mb) vector
%       n = number of rows in input
%       (:,1,1) -> firing strength of each rule
%
% @param input 1 -> membership_values
%
%       (n,m) vector
%       n = number of input membership
%       function or number of outputs
%       m = number of inputs to FIS system or number of features
%       (:,1) -> fuzzified values of input one for each membership function
%       (1,:) -> fuzzified output of membership fuction 1 of each input
%
%

output = prod(membership_values,2);


end