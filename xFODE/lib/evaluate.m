function yPreds_mean = evaluate(xTrain0, t, ux, Learnable_parameters, number_inputs, mbs, number_of_rules, number_outputs, input_membership_type)

x0 = xTrain0;
yPred = x0';
ahead = length(t)-1;
Ux = permute(ux, [1, 3, 2]);
X_mean = dlarray(zeros(size(x0,2) ,ahead, size(x0,1)));

for ct=1:ahead

    subnet_outputs = [];

    u0 = permute(Ux(:, :, ct), [3 1 2]);
    z0 = cat(2, x0, u0);

    for i = 1:number_inputs
        subnet = Learnable_parameters.("subnet" + i);
        subnet_output = fismodel(z0(:, i, :), number_of_rules, 1, number_outputs, mbs, subnet, input_membership_type);
        subnet_output = permute(subnet_output, [1 3 2]);
        subnet_outputs = [subnet_outputs;subnet_output];
    end
    dx = aggregration_output(subnet_outputs);

    x_new = x0 + dx;
    x_new = permute(x_new, [2, 1, 3]);
    X_mean(:, ct, :) = x_new;
    x0 = permute(x_new, [2, 1, 3]);

end

yPreds_mean = [yPred X_mean];

end

