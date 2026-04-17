
function output = custom_gauss2mf(x, left_sigma, right_sigma, c)
    % Custom Gaussian Membership Function with asymmetric (left-right) sigmas
    % x           -> input values [5, 4, batch size]
    % left_sigma  -> sigma to the left of the center [5, 4]
    % right_sigma -> sigma to the right of the center [5, 4]
    % c           -> center of the Gauss MF [5, 4]
    % 
    % Expand left_sigma, right_sigma, and c to match x dimensions
    left_sigma = repmat(left_sigma, 1, 1, size(x, 3));
    right_sigma = repmat(right_sigma, 1, 1, size(x, 3));
    c = repmat(c, 1, 1, size(x, 3));

    % Initialize output array
    output = dlarray(zeros(size(x)));

    % Calculate membership values based on left and right sigmas
    left_indices = x <= c;
    right_indices = x > c;

    % Left side calculation (x <= c)
    output(left_indices) = exp(-0.5 * ((x(left_indices) - c(left_indices)).^2 ./ left_sigma(left_indices).^2));

    % Right side calculation (x > c)
    output(right_indices) = exp(-0.5 * ((x(right_indices) - c(right_indices)).^2 ./ right_sigma(right_indices).^2));


end