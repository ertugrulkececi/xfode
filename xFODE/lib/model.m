function X = model(t, mini_batch_inputs, u_mini_batch, number_of_rule, number_inputs, number_outputs, mbs, learnable_parameters, input_membership_type)

x0 = mini_batch_inputs;
x = x0;

ahead = length(t)-1;
Ux = permute(u_mini_batch, [1, 3, 2]);
X = dlarray(zeros(size(x,2) ,ahead, size(x,3)));

for ct = 1:ahead

    subnet_outputs = [];

    u0 = Ux(:, :, ct);
    x_ = permute(x, [2 3 1]);
    z0 = permute([x_;u0], [3 1 2]);

    for i = 1:number_inputs
        subnet = learnable_parameters.("subnet" + i);
        subnet_output = fismodel(z0(:, i, :), number_of_rule, 1, number_outputs, mbs, subnet, input_membership_type);
        subnet_output = permute(subnet_output, [1 3 2]);
        subnet_outputs = [subnet_outputs;subnet_output];
    end

    dx = aggregration_output(subnet_outputs);

    x_new = x + dx;
    x_new = permute(x_new, [2, 1, 3]);
    X(:, ct, :) = x_new;
    x = permute(x_new, [2, 1, 3]);

end

end

