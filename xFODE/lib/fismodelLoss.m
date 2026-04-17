function [loss, gradients, yPred] = fismodelLoss(timesteps, mini_batch_inputs, number_inputs, u_mini_batch, targets, number_outputs, number_of_rules, mbs, learnable_parameters, input_membership_type)

yPred = model(timesteps, mini_batch_inputs, u_mini_batch, number_of_rules, number_inputs, number_outputs, mbs, learnable_parameters, input_membership_type);

loss = l1loss(yPred, targets, NormalizationFactor="batch-size", DataFormat="STB");

gradients = dlgradient(loss, learnable_parameters);

end