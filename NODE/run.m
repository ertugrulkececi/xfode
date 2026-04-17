clear;clc

addpath(fullfile(pwd,'lib'));
parallel.gpu.enableCUDAForwardCompatibility(true)
%% For reproducability

number_of_runs = 20;
dataset_name = "MRDamper"; %MRDamper, HairDryer, TwoTank, EVBattery, SteamEngine
SR_method = "incremental"; % SR1: lagged, SR2: incremental
loss_type = "L2R";
[number_of_epoch, learnRate, mbs, neuralOdeTimesteps, lag, Input, Output] = training_prep(dataset_name);
[xTrain, xTrain0, yTrue, xTest, xTest0, uTest, tTest, training_num, t, std1, mu1, ny] = data_prep(dataset_name, SR_method, lag);


%% Training

numExperiment = height(xTrain) - neuralOdeTimesteps;
Expts = cell(1, numExperiment); 

Time = seconds(0.1:(size(xTrain,1)-0.9));
data = array2timetable(xTrain,'RowTimes',Time);

for i = 1:numExperiment
    Expts{i} = data(i:i+neuralOdeTimesteps,:);
    if i>1
       % set the row time of each segment to be identical; this is a requirement for training a
       % neural state-space model with multiple data experiments
       Expts{i}.Properties.RowTimes = Expts{1}.Properties.RowTimes;
    end
end

lr = learnRate;

Ts = 1;


%%

opt = nssTrainingOptions('adam');
opt.MaxEpochs = number_of_epoch;
opt.MiniBatchSize = mbs;
opt.PlotLossFcn = false;

opt.LearnRate = lr;


seed_list = linspace(0, number_of_runs-1, number_of_runs);
all_RMSE   = zeros(size(yTrue,1), number_of_runs);

for seed = seed_list
    rng(seed)

    clear nss

    nss = idNeuralStateSpace(size(Output,2),NumInputs=size(Input,2), Ts=Ts);
    nss.InputName = Input;%added
    nss.OutputName = Output;%added


    nss.StateNetwork = createMLPNetwork(nss,'state', ...
        LayerSizes= [128 128], ...
        WeightsInitializer="glorot", ...
        BiasInitializer="zeros", ...
        Activations='tanh');

    nss = nlssest(Expts,nss,opt);%added

    U = array2timetable(uTest',RowTimes=seconds(tTest'), VariableNames=Input);%added

    % Simulate neural state-space system from x0
    simOpt = simOptions('InitialCondition',xTest0');
    yn = sim(nss,U,simOpt);

    yPred = table2array(yn)';
    yPreds_mean = yPred.*std1 + mu1;
    yTestVal = xTest.*std1 + mu1;


    err = yTestVal - yPreds_mean;

    NRMSE = sqrt(sum(err.^2, 2)./sum((yTestVal-mean(yTestVal, 2)).^2, 2));
    accuracy = 100*(1-NRMSE);

    testRMSE = rmse(yTestVal, yPreds_mean, 2);

    run_idx = seed + 1;
    all_RMSE(:, run_idx)     = testRMSE;
end


%% Summary

fprintf('\n--- Results over %d runs (%s | %s) ---\n', ...
    number_of_runs, dataset_name, SR_method);
for o = 1:ny
    fprintf('Output %d | RMSE:     mean = %.4f,  std = %.4f\n', o, mean(all_RMSE(o,:)),     std(all_RMSE(o,:)));
end








