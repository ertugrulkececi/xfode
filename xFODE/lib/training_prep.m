function [number_of_epoch, learnRate, mbs, neuralOdeTimesteps, lag] = training_prep(dataset_name)

if dataset_name == "HairDryer"

    neuralOdeTimesteps = 20;
    mbs = 64;
    number_of_epoch = 200;
    learnRate = 0.002;
    lag = 2;

elseif dataset_name == "TwoTank"
    neuralOdeTimesteps = 20;
    mbs = 256;
    number_of_epoch = 200;
    learnRate = 0.002;
    lag = 2;

elseif dataset_name == "EVBattery"
    neuralOdeTimesteps = 20;
    mbs = 512;
    number_of_epoch = 200;
    learnRate = 0.01;
    lag = 1;

elseif dataset_name == "SteamEngine"
    neuralOdeTimesteps = 20;
    mbs = 32;
    number_of_epoch = 200;
    learnRate = 0.01;
    lag = 1;

elseif dataset_name == "MRDamper"
    neuralOdeTimesteps = 20;
    mbs = 256;
    number_of_epoch = 200;
    learnRate = 0.01;
    lag = 2;

end

end
