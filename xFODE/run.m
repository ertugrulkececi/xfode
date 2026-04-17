clc;clear;
close all;

addpath(fullfile(pwd,'lib'));
parallel.gpu.enableCUDAForwardCompatibility(true)
%% Training Options

number_of_rules = 5;
number_of_runs = 20;
SR_method = "incremental"; % SR1: lagged, SR2: incremental
dataset_name = "EVBattery"; %MRDamper, HairDryer, TwoTank, EVBattery, SteamEngine
[number_of_epoch, learnRate, mbs, neuralOdeTimesteps, lag] = training_prep(dataset_name);

%% Data Load

[xTrain, xTrain0, yTrue, xTest, xTest0, uTest, tTest, training_num, t, std1, mu1, ny] = data_prep(dataset_name, SR_method, lag);

number_inputs = size(xTrain,2);
number_outputs = size(yTrue, 1);

%% FLS Configuration

input_membership_type = "trimf"; % xFODE-PS1: trimf, xFODE-PS2: gauss2mf, xFODE-PS3: c-gauss2mf, AFODE: gaussmf 

%% Training Loop
gradDecay = 0.9;
sqGradDecay = 0.999;

seed_list = linspace(0, number_of_runs-1, number_of_runs);
all_RMSE   = zeros(number_outputs, number_of_runs);

for seed = seed_list
    rng(seed)

    clear gradients

    averageGrad = [];
    averageSqGrad = [];

    data = [xTrain];

    [Train] = split_data(data, number_inputs, number_outputs, training_num);

    [Learnable_parameters, timesteps] = initialize_model(Train.inputs, Train.outputs, number_of_rules, input_membership_type, t, neuralOdeTimesteps);

    X = Train.inputs(:, 1:number_outputs, :);
    ux = Train.inputs(:, number_outputs+1:end, :);
    ux = permute(ux, [2 3 1]);
    ux = extractdata(ux);

    rng(seed)

    number_of_iter_per_epoch = floorDiv(training_num - neuralOdeTimesteps, mbs);

    for epoch = 1:number_of_epoch

        [batch_inputs, U, batch_targets] = create_mini_batch(X, ux, neuralOdeTimesteps, training_num-neuralOdeTimesteps);

        batch_loss = 0;

        for iter = 1:number_of_iter_per_epoch

            [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets, iter, mbs);

            [loss, gradients, ~] = dlfeval(@fismodelLoss, timesteps, mini_batch_inputs, number_inputs, u_mini_batch, targets, number_outputs, number_of_rules, mbs, Learnable_parameters, input_membership_type);

            [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad, ...
                epoch, learnRate, gradDecay, sqGradDecay);

            batch_loss = batch_loss + loss;

        end

        batch_loss = batch_loss / number_of_iter_per_epoch;
        
        % fprintf('Seed %d | Epoch %d | Batch_Loss = %.4f\n', seed, epoch, batch_loss); 

    end

    %% Inference

    yPreds_mean = evaluate(xTest0, tTest, uTest, Learnable_parameters, number_inputs, mbs, number_of_rules, number_outputs, input_membership_type);

    yPreds_mean = yPreds_mean .* std1 + mu1;
    yTestVal    = xTest .* std1 + mu1;

    err = yTestVal - yPreds_mean;

    NRMSE    = sqrt(sum(err.^2, 2) ./ sum((yTestVal - mean(yTestVal, 2)).^2, 2));
    testRMSE = rmse(yTestVal, yPreds_mean, 2);

    run_idx = seed + 1;
    all_RMSE(:, run_idx)     = extractdata(testRMSE);

end

%% Summary

fprintf('\n--- Results over %d runs (%s | %s | %s) ---\n', ...
    number_of_runs, dataset_name, SR_method, input_membership_type);
for o = 1:ny
    fprintf('Output %d | RMSE:     mean = %.4f,  std = %.4f\n', o, mean(all_RMSE(o,:)),     std(all_RMSE(o,:)));
end