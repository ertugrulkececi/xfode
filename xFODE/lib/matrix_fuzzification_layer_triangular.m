function [output] = matrix_fuzzification_layer_triangular(x, learnable_parameters, number_of_rules)

if nargin<4
    min_x = 0;
    max_x = 1;
    eps= 0.1;
end

if isa(x, "double")
    x = dlarray(x);
end

lambdas = log(1+ exp(learnable_parameters.lambdas));
% lambdas = abs(learnable_parameters.lambdas);

% params = [learnable_parameters.left(1, :), learnable_parameters.lambdas'];
% params = [learnable_parameters.left(1, :), lambdas'];
params = [learnable_parameters.leftmost_center(1, :), lambdas(2:end-1,:)'];
T = dlarray(tril(ones(length(params))));
T = T.* params;

c = T* dlarray(ones(length(params), 1));

l = [-1e6; c(1:end-1)];
r = [c(2:end);1e6];






% l(1, :) = -1e6;
% 
% c(1, :) = learnable_parameters.left(1, :) + abs(learnable_parameters.lambdas(1, :));
% r(1, :) = c(1, :) + abs(learnable_parameters.lambdas(2, :));
% 
% 
% for i = 2:number_of_rules-1
%     l(i, :) = c(i-1, :);
%     c(i, :) = r(i-1, :);
%     r(i, :) = c(i, :) + abs(learnable_parameters.lambdas(i+1, :));
% end
% 
% l(number_of_rules, :) = c(number_of_rules-1, :);
% c(number_of_rules, :) = r(number_of_rules-1, :);
% r(number_of_rules, :) = 1e6;



output = custom_triangular_mf(x, l, c, r);


end
%%

% Custom Triangular function
function output = custom_triangular_mf(x, l, c, r)

    output = max(min((x-l)./(c-l), (r-x)./(r-c)), 0);

end