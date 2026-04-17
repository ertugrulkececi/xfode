
function output = custom_c_gauss2mf(x, left_sigma, right_sigma, c)
    % Custom Gaussian Membership Function with asymmetric (left-right) sigmas
    % x           -> input values [5, 4, batch size]
    % left_sigma  -> sigma to the left of the center [5, 4]
    % right_sigma -> sigma to the right of the center [5, 4]
    % c           -> center of the Gauss MF [5, 4]
    % 
    %     Initialize output array
    output = dlarray(zeros(size(x)));

    m = size(x,1);
    B = size(x,3);

    maskL   = (x <= c);
    gaussL = exp(-0.5 * ((x - c).^2) ./ left_sigma.^2);
    gaussR = exp(-0.5 * ((x - c).^2) ./ right_sigma.^2);
    evenAll = gaussL .* maskL + gaussR .* ~maskL;

    evenSel = mod((1:m).',2)==0;
    evenOut = evenAll .* evenSel; 

    cPrev = [-inf; c(1:end-1)];
    cNext = [c(2:end); inf];

    inLeftInt  = (cPrev < x) & (x < c);   % (c_{i-1}, c_i)
    inRightInt = (c < x)    & (x < cNext);% (c_i, c_{i+1})

    prevEven = cat(1, zeros(1,1,B,'like',x), evenAll(1:end-1,:,:)); % i-1
    nextEven = cat(1, evenAll(2:end,:,:),     zeros(1,1,B,'like',x)); % i+1

    oddRaw = inLeftInt .* (1 - prevEven) + inRightInt .* (1 - nextEven);

    oddSel = mod((1:m).',2)==1;

    oddOut = oddRaw .* oddSel;

    output = evenOut + oddOut;

end