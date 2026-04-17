function [number_of_epoch, learnRate, mbs, neuralOdeTimesteps, lag, Input, Output] = training_prep(dataset_name)

if dataset_name == "HairDryer"

    neuralOdeTimesteps = 20;
    mbs = 64;
    number_of_epoch = 200;
    learnRate = 0.0002;
    lag = 2;
    Input = ["xTrain4"];
    Output = ["xTrain1", "xTrain2", "xTrain3"];

elseif dataset_name == "TwoTank"
    neuralOdeTimesteps = 20;
    mbs = 256;
    number_of_epoch = 200;
    learnRate = 0.0002;
    lag = 2;
    Input = ["xTrain4"];
    Output = ["xTrain1", "xTrain2", "xTrain3"];

elseif dataset_name == "EVBattery"
    neuralOdeTimesteps = 20;
    mbs = 512;
    number_of_epoch = 200;
    learnRate = 0.001;
    lag = 1;
    Input = ["xTrain3", "xTrain4"];
    Output = ["xTrain1", "xTrain2"];

elseif dataset_name == "SteamEngine"
    neuralOdeTimesteps = 20;
    mbs = 32;
    number_of_epoch = 200;
    learnRate = 0.001;
    lag = 1;
    Input = ["xTrain5", "xTrain6"];
    Output = ["xTrain1", "xTrain2", "xTrain3", "xTrain4"];

elseif dataset_name == "MRDamper"
    neuralOdeTimesteps = 20;
    mbs = 256;
    number_of_epoch = 200;
    learnRate = 0.001;
    lag = 2;
    Input = ["xTrain4"];
    Output = ["xTrain1", "xTrain2", "xTrain3"];
end

end
