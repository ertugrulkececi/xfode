function [xTrain, xTrain0, yTrue, xTest, xTest0, uTest, tTest, training_num, t, std1, mu1, ny] = data_prep(dataset_name, SR_method, lag)

if dataset_name == "HairDryer"

    load dryer2

    dry = iddata(y2,u2,0.08);

    dry.InputName = 'Heater Voltage';
    dry.OutputName = 'Thermocouple Voltage';
    dry.TimeUnit = 'seconds';
    dry.InputUnit = 'V';
    dry.OutputUnit = 'V';

    ze = dry(1:500);
    ze = detrend(ze);
    zv = dry(501:end);
    zv = detrend(zv);

    datatrain = [ze.OutputData ze.InputData];
    datatest = [zv.OutputData zv.InputData];

    eDataN = datatrain;
    vDataN = datatest;

    Ts = 1;

    ny = 1;

    xTrain = input_shaping(eDataN, lag, SR_method);

    c_x = mean(xTrain);
    s_x = std(xTrain, 0, 1);
    xTrain = (xTrain - c_x) ./ s_x;

    xTrain0 = xTrain(1, 1:lag+1);
    yTrue = xTrain(:, 1:lag+1)';

    training_num = size(xTrain, 1);
    t = 0.1:Ts:size(xTrain, 1)*Ts;

    xTest = input_shaping(vDataN, lag, SR_method);
    xTest = (xTest - c_x) ./ s_x;

    xTest0 = xTest(1, 1:lag+1);
    uTest  = permute(xTest(:, lag+2:end), [2 1]);
    tTest  = 0.1:Ts:size(xTest, 1)*Ts;
    xTest  = xTest(:, 1:lag+1)';

    std1 = s_x(1:lag+1)';
    mu1  = c_x(1:lag+1)';

elseif dataset_name == "TwoTank"
    load twotankdata

    z1f = iddata(y, u, 0.2, 'Name', 'Two-tank system');

    datatrain = [z1f(1:1500).OutputData    z1f(1:1500).InputData];
    datatest  = [z1f(1501:3000).OutputData z1f(1501:3000).InputData];

    Ts = 1;

    ny = 1;

    xTrain = input_shaping(datatrain, lag, SR_method);

    c_x = mean(xTrain);
    s_x = std(xTrain, 0, 1);
    xTrain = (xTrain - c_x) ./ s_x;

    xTrain0 = xTrain(1, 1:lag+1);
    yTrue = xTrain(:, 1:lag+1)';

    training_num = size(xTrain, 1);
    t = 0.1:Ts:size(xTrain, 1)*Ts;

    xTest = input_shaping(datatest, lag, SR_method);
    xTest = (xTest - c_x) ./ s_x;

    xTest0 = xTest(1, 1:lag+1);
    uTest  = permute(xTest(:, lag+2:end), [2 1]);
    tTest  = 0.1:Ts:size(xTest, 1)*Ts;
    xTest  = xTest(:, 1:lag+1)';

    std1 = s_x(1:lag+1)';
    mu1  = c_x(1:lag+1)';

elseif dataset_name == "EVBattery"
    load(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'datasets', 'EVBattery', 'EVBatteryTemperature.mat'))

    datatrain = [datae.y_temp datae.u_curr datae.u_soc];
    datatest = [datav.y_temp datav.u_curr datav.u_soc];

    ny = 1;
    nx = (lag+1)*ny;
    
    Ts = 1;

    xTrain = input_shaping(datatrain, lag, SR_method, ny);
    c_x = mean(xTrain, 1);
    s_x = std(xTrain, 0, 1);
    xTrain = (xTrain - c_x) ./ s_x;

    xTrain0 = xTrain(1, 1:nx);
    yTrue   = xTrain(:, 1:nx)';
    training_num = size(xTrain, 1);
    t = 0.1:Ts:training_num*Ts;

    xTest = input_shaping(datatest, lag, SR_method, ny);
    xTest = (xTest - c_x) ./ s_x;

    xTest0 = xTest(1, 1:nx);
    uTest  = permute(xTest(:, nx+1:end), [2 1]);
    tTest  = 0.1:Ts:size(xTest, 1)*Ts;
    xTest  = xTest(:, 1:nx)';

    std1 = s_x(1:nx)';
    mu1  = c_x(1:nx)';

elseif dataset_name == "SteamEngine"
    load SteamEng

    data = [GenVolt Speed Pressure MagVolt];

    datatrain = data(1:250,:);
    datatest  = data(251:end,:);

    ny = 2;
    nx = (lag+1)*ny;
    Ts = 1;

    xTrain = input_shaping(datatrain, lag, SR_method, ny);
    c_x = mean(xTrain, 1);
    s_x = std(xTrain, 0, 1);
    xTrain = (xTrain - c_x) ./ s_x;

    xTrain0 = xTrain(1, 1:nx);
    yTrue   = xTrain(:, 1:nx)';
    training_num = size(xTrain, 1);
    t = 0.1:Ts:training_num*Ts;

    xTest = input_shaping(datatest, lag, SR_method, ny);
    xTest = (xTest - c_x) ./ s_x;

    xTest0 = xTest(1, 1:nx);
    uTest  = permute(xTest(:, nx+1:end), [2 1]);
    tTest  = 0.1:Ts:size(xTest, 1)*Ts;
    xTest  = xTest(:, 1:nx)';

    std1 = s_x(1:nx)';
    mu1  = c_x(1:nx)';

elseif dataset_name == "MRDamper"  
    load mrdamper

    data = [F V];

    datatrain = data(1:3000,:);
    datatest = data(3001:end,:);

    Ts = 1;

    ny = 1;

    xTrain = input_shaping(datatrain, lag, SR_method);
    c_x = mean(xTrain);
    s_x = std(xTrain, 0, 1);
    xTrain = (xTrain - c_x) ./ s_x;

    xTrain0 = xTrain(1, 1:lag+1);
    yTrue = xTrain(:, 1:lag+1)';
    training_num = size(xTrain, 1);
    t = 0.1:Ts:size(xTrain, 1)*Ts;

    xTest = input_shaping(datatest, lag, SR_method);
    xTest = (xTest - c_x) ./ s_x;

    xTest0 = xTest(1, 1:lag+1);
    uTest  = permute(xTest(:, lag+2:end), [2 1]);
    tTest  = 0.1:Ts:size(xTest, 1)*Ts;
    xTest  = xTest(:, 1:lag+1)';

    std1 = s_x(1:lag+1)';
    mu1  = c_x(1:lag+1)';
end


end
