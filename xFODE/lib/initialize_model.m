function [subnets, timesteps] = initialize_model(input_data, output_data, number_of_rule, input_membership_type, t, neuralOdeTimesteps)

dt = t(2) - t(1);
timesteps = (0:neuralOdeTimesteps)*dt + t(1);

number_inputs = size(input_data, 2);

subnets = struct;

for i = 1:number_inputs
    if input_membership_type == "gauss2mf" || input_membership_type == "c-gauss2mf"
        subnet = initialize_homogenus_sigmas(input_data(:, i, :), output_data, number_of_rule);
    elseif input_membership_type == "gaussmf" 
        subnet = initialize_Glorot_Kmeans(input_data(:, i, :), output_data, number_of_rule);
    elseif input_membership_type == "trimf"
        subnet = initialize_trimf(input_data(:, i, :), output_data, number_of_rule);
    end
    subnets.("subnet" + i) = subnet;
end

end
