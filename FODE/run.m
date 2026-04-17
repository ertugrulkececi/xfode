clear;clc

addpath(fullfile(pwd,'lib'));
parallel.gpu.enableCUDAForwardCompatibility(true)
%% For reproducability

number_mf = 5;
number_of_runs = 20;
dataset_name = "TwoTank"; %MRDamper, HairDryer, TwoTank, EVBattery, SteamEngine
SR_method = "incremental"; % SR1: lagged, SR2: incremental
loss_type = "L2R";
[number_of_epoch, learnRate, mbs, neuralOdeTimesteps, lag] = training_prep(dataset_name);
[xTrain, xTrain0, yTrue, xTest, xTest0, uTest, tTest, training_num, t, std1, mu1, ny] = data_prep(dataset_name, SR_method, lag);


%%

number_inputs = size(xTrain,2); % total column space, nx + nu
number_outputs = size(yTrue,1); % 

input_membership_type = "gaussmf";

output_membership_type = "linear";


gradDecay = 0.9;
sqGradDecay = 0.999;

close all
%%

seed_list = linspace(0, number_of_runs-1, number_of_runs);
all_RMSE   = zeros(number_outputs, number_of_runs);

for seed = seed_list
    rng(seed)

    clear gradients

    averageGrad = [];
    averageSqGrad = [];

    % split by number ------------------------------
    data = [xTrain];
    data_size = size(data,1);
    test_num = data_size-training_num;


    Training_temp = data((1:training_num),:);

    % ------------------------------

    %training data
    Train.inputs = reshape(Training_temp(:,1:number_inputs)', [1, number_inputs, training_num]); % traspose come from the working mechanism of the reshape, so it is a must
    Train.outputs = reshape(Training_temp(:,1:number_outputs)', [1, number_outputs, training_num]);

    Train.inputs = dlarray(Train.inputs);

    % init

    [Learnable_parameters, timesteps] = initialize_Glorot_Kmeans(Train.inputs, Train.outputs, number_mf, output_membership_type, t, neuralOdeTimesteps);
    prev_learnable_parameters = Learnable_parameters;

    % data manupulation

    X = Train.inputs(:, 1:number_outputs, :);
    ux = Train.inputs(:, number_outputs+1:end, :);
    ux = permute(ux, [2 3 1]);
    ux = extractdata(ux);


    % rng reset
    rng(seed)

    number_of_iter_per_epoch = floorDiv(training_num-neuralOdeTimesteps, mbs);

    number_of_iter = number_of_epoch * number_of_iter_per_epoch;
    global_iteration = 1;

    for epoch = 1: number_of_epoch

        [batch_inputs, U, batch_targets] = create_mini_batch(X, ux, neuralOdeTimesteps, training_num-neuralOdeTimesteps);

        batch_loss = 0;

        for iter = 1:number_of_iter_per_epoch

            [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets, iter, mbs);

            [loss, gradients, yPred_train] = dlfeval(@NODEfisModelLoss_u, timesteps, mini_batch_inputs, number_inputs, u_mini_batch, targets,number_outputs, number_mf, mbs, Learnable_parameters, output_membership_type);


            [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
                epoch, learnRate, gradDecay, sqGradDecay);

            batch_loss = batch_loss + loss;

        end

        batch_loss = batch_loss/number_of_iter_per_epoch;

        % fprintf('Seed %d | Epoch %d | Batch_loss = %.4f\n', seed, epoch, batch_loss);

    end


    % Inference-discrete

    x0 = xTest0;
    yPred = x0';
    ahead = length(tTest)-1;
    PP = griddedInterpolant(tTest, permute(uTest,[2, 1, 3]), "pchip");
    Ux = permute(PP(tTest(:)),[2 3 1]);
    X_mean = dlarray(zeros(size(x0,2) ,ahead, size(x0,1)));

    for ct = 1:ahead
        u = Ux(:,:,ct);
        x = fismodel_u(x0, u, number_mf, number_inputs,number_outputs, mbs, Learnable_parameters, output_membership_type);
        X_mean(:, ct, :) = x';
        x0 = x;
    end

    yPreds_mean = [yPred X_mean];

    yPreds_mean = yPreds_mean.*std1 + mu1;
    yTestVal = xTest.*std1 + mu1;

    err = yTestVal - yPreds_mean;

    NRMSE = sqrt(sum(err.^2, 2)./sum((yTestVal-mean(yTestVal, 2)).^2, 2));
    accuracy = 100*(1-NRMSE);


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
%%
function [X0, U, targets]  = create_mini_batch(X, ux,  ahead, numexamples)

X = permute(X, [2, 3, 1]);

shuffle_idx = randperm(size(X, 2)-ahead);

X0 = dlarray(X(:, shuffle_idx));
targets = dlarray(zeros([size(X, 1) ahead, numexamples]));
U = (zeros([size(ux, 1), ahead+1, numexamples]));

for i =1:numexamples
    targets(:, :, i) = X(:, shuffle_idx(i) + 1: shuffle_idx(i) + ahead);
     U(:, :, i) = ux(:, shuffle_idx(i): shuffle_idx(i) + ahead);
end

X0 = permute(X0, [3 1 2]);

end


%%
function [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets,iter,mbs)

mini_batch_inputs = batch_inputs(:, :, ((iter-1)*mbs)+1:(iter*mbs));
targets = batch_targets(:, :, ((iter-1)*mbs)+1:(iter*mbs));
u_mini_batch = U(:, :, ((iter-1)*mbs)+1:(iter*mbs));

end



