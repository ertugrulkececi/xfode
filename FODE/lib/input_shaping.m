function [X] = input_shaping(x, order, SR_method, nx)
% Unified input shaping for single- and multi-output datasets.
%
%   x         : [N x (nx + nu)] raw data matrix, outputs first then inputs
%   order     : number of lags / difference orders
%   SR_method : "lagged" (SR1) or "incremental" (SR2)
%   nx        : number of output columns in x (default: 1)
%
% Output X : [N-order x ((order+1)*nx + nu)] regressor matrix

if nargin < 4
    nx = 1;
end

y  = x(:, 1:nx);
u  = x(:, nx+1:end);
N  = size(x, 1);
ny = nx;
nu = size(u, 2);

idx   = (order+1:N).';
nrows = numel(idx);

Ycols = zeros(nrows, (order+1)*ny);
Ucols = u(idx, :);

if SR_method == "lagged"
    for j = 0:order
        Ycols(:, j*ny + (1:ny)) = y(idx - j, :);
    end
elseif SR_method == "incremental"
    Ycols(:, 1:ny) = y(idx, :);
    yj = y;
    for j = 1:order
        yj = yj(2:end, :) - yj(1:end-1, :);
        Ycols(:, j*ny + (1:ny)) = yj(idx - j, :);
    end
end

X = [Ycols, Ucols];

end

