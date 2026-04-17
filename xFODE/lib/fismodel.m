function ypred = fismodel(mini_batch_inputs, number_of_rule, number_inputs, number_outputs, mbs, learnable_parameters, input_membership_type)

    fuzzified = matrix_fuzzification_layer(mini_batch_inputs, input_membership_type, learnable_parameters, number_of_rule, number_inputs, mbs);
    firestrength = firing_strength_calculation_layer(fuzzified);
    normalized = firing_strength_normalization_layer(firestrength);
    ypred = multioutput_defuzzification_layer(mini_batch_inputs, normalized, learnable_parameters, number_outputs);

end


















