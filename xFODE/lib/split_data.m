function [Train] = split_data(data, number_inputs, number_outputs, training_num)

    data_size = height(data);
    % training_num = round(data_size*fracTrain*0.8);
    % validation_num = round(data_size*fracTrain*0.2);
    % test_num = data_size - (training_num + validation_num);
    test_num = data_size - training_num;

    % if shuffle == "true"
    %     idx = randperm(data_size);
    % else
    %     idx = 1:data_size;
    % end



    Training_temp = data((1:training_num),:);
    % Validation_temp = data((training_num + 1 : training_num + validation_num), :);
    % Testing_temp = data((training_num + validation_num +1:end),:);

    %training data
    Train.inputs = reshape(Training_temp(:,1:number_inputs)', [1, number_inputs, training_num]); % traspose come from the working mechanism of the reshape, so it is a must
    Train.outputs = reshape(Training_temp(:,1:number_outputs)', [1, number_outputs, training_num]);

    Train.inputs = dlarray(Train.inputs);
    Train.outputs = dlarray(Train.outputs);


end