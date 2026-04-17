function output = multioutput_defuzzification_layer(x,normalized_firing_strength, learnable_parameters,number_outputs)


%preparing the input and slopes and biases of linear consequent to
%multipication
temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b]; % adding the bias to the end of slope matrix
x = permute(x,[2 1 3]);
temp_input = [x; ones(1, size(x, 2), size(x, 3))]; % Append a row of ones to the input for bias multiplication.
temp_input = permute(temp_input, [1 3 2]); % Permute input to align dimensions for further operations.

% Multiply the membership function parameters with the input.
c = temp_mf*temp_input;
% Reshape the result to align with the normalized firing strengths.
c = reshape(c, [size(normalized_firing_strength, 1), number_outputs, size(normalized_firing_strength, 3)]);

% Replicate normalized firing strengths for each output.
normalized_firing_strength = repmat(normalized_firing_strength,1,number_outputs);

% Multiply normalized firing strengths elementwise with the result of linear operation.
output = normalized_firing_strength.* c;
output = sum(output, 1); % Sum the values across the first dimension to get the final output.
output = dlarray(output);

end