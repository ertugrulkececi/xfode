function y_pred = aggregration_output(subnet_outputs)

y_pred = sum(subnet_outputs, 1);

y_pred = permute(y_pred, [1 3 2]);

end