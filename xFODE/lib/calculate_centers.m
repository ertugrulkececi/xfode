function [centers, left_sigmas, right_sigmas] = calculate_centers(leftmost_center, sigmas)
    % Calculate the Gaussian centers and left/right sigmas.
    % leftmost_center: The first center, scalar dlarray.
    % sigmas: dlarray of shape [m+1, n], where:
    %         - sigma(1, :) is the left sigma of the first Gaussian,
    %         - sigma(end, :) is the right sigma of the last Gaussian.
    %         - Each middle sigma is used for calculating centers.

    % Number of centers (m) is one less than the length of sigmas
    m = size(sigmas, 1) - 1;
    n = size(sigmas, 2); % Number of columns for different batches/experiments
    % sigmas = abs(sigmas);
    sigmas = log(1 + exp(sigmas));
    % Calculate the left and right sigmas for each Gaussian
    left_sigmas = sigmas(1:end-1, :);
    right_sigmas = sigmas(2:end, :);

    % % Initialize centers matrix
    % centers = dlarray(zeros(m, n), 'CB'); % Assuming batch mode for the second dimension
    % 
    % % Set the first center based on leftmost_center
    % centers(1, :) = leftmost_center;
    % 
    % % Calculate each subsequent center
    % for i = 2:m
    %     centers(i, :) = centers(i - 1, :) + 4 * sigmas(i, :);
    % end

    %% Alternative center calculation
    % Extract the middle sigmas and multiply by 3 to get increments
    increments = 4 * sigmas(2:end-1, :);
    params = [leftmost_center, increments'];
    T = dlarray(tril(ones(length(params))));
    T = T.* params;
    centers = T* dlarray(ones(length(params), 1));
    
    % % Create a cumulative sum matrix to calculate displacements
    % cumulative_matrix = tril(ones(m-1));
    % 
    % % Calculate all displacements in one step
    % displacements = cumulative_matrix * increments;
    % 
    % % Expand leftmost_center to match batch size
    % leftmost_center_expanded = repmat(leftmost_center, 1, n);
    % 
    % % Calculate centers by adding displacements to the leftmost center
    % centers = dlarray([leftmost_center_expanded; leftmost_center_expanded + displacements], 'CB');

    % Limit multipliers to the range [0.5, 1] with sigmoid function
    % multipliers = 0.5 + 0.5 * sigmoid(multipliers);
    % % multipliers = 0.1 + 0.9 * sigmoid(multipliers);
    % 
    % % Enforce the condition: left_multipliers >= 1 - right_multipliers, order is like this: start from second left and first right, and go 1 by 1
    % for i = 2:m
    %     multipliers(i,:,1) = max(multipliers(i,:,1), 1 - multipliers(i-1,:,2));
    % end
    % % multipliers(:,:,1) = (1 - multipliers(:,:,2)) + log(1 + exp(multipliers(:,:,1) - multipliers(:,:,2)));
    % 
    % Apply multipliers to the left and right sigmas but don't multiply the first sigmas of left_sigmas and last sigmas of right_sigmas
    % left_sigmas(2:end, :) = left_sigmas(2:end, :) .* multipliers(2:end, :,1);
    % right_sigmas(1:end-1, :) = right_sigmas(1:end-1, :) .* multipliers(1:end-1, :,2);
    % % 
    left_sigmas(1, :, :) = left_sigmas(1, :, :) * 1e6;    % Leftmost left sigma
    right_sigmas(end, :, :) = right_sigmas(end, :, :) * 1e6;  % Rightmost right sigma


end