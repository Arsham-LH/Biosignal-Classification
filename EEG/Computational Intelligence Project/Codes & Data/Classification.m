%% Load Features matrix
clc;
close all;
clear;

load('testMat.mat');
load('trainMat.mat');
Labels = cell2mat(struct2cell(load('TrainLabels.mat')));

%********Shuffling Features**************
% % shuff_idx = randperm(size(trainMat,1));
% load('shuff_idx.mat');
% trainMat = trainMat(shuff_idx, :);
% testMat = testMat(shuff_idx, :);

% % save('shuff_idx.mat', 'shuff_idx');
%****************************************

%% Classification: MLP
clc;

n = size(trainMat, 2); %Total number of training and validation points
val_n = 0.2*n; %Number of validation points

maxHiddens = 10; %The maximum number of hidden layers
maxNeurons = 8; %The maximum number of neurons in the hidden layer
trainsNum = 5; %The number of trainings for each maxNeurons

best_acc = 0;
best_hiddens = 0; %Best number of Hidden layers
best_neurons = 0; %Best number of neurons per hidden layer
for i = 1:maxHiddens
    disp("Hidden Layer: "+i);
    for j = 1:maxNeurons
        disp("Neurons: "+j);

        val_acc_arr = zeros(5, 1); %Accuracy values for each training (on validation data points)
        for k = 1:5
            %******************Set validation points here:*******************
            valPoints_ind = (k-1)*val_n+1 : k*val_n;
            ValData = trainMat(:, valPoints_ind); %Validation data
            TrainData = trainMat; TrainData(:, valPoints_ind) = []; %Training Data

            ValLabels = Labels(valPoints_ind);
            TrainLabels = Labels; TrainLabels(valPoints_ind) = [];
            %****************************************************************

            %Creating the network:
            net = feedforwardnet(repmat(j, 1, i), 'traingd');
            net.trainParam.epochs = 500; % Number of training epochs
            net.divideParam.trainRatio = 1;
            net.divideParam.testRatio = 0;
            net.divideParam.valRatio = 0;


            pred_y_arr = zeros(trainsNum, val_n); %Predicted outputs for each training

            for tr = 1:trainsNum
                trained_net = train(net, TrainData, TrainLabels); %Training the network using training data points
                pred_y_arr(tr, :) = trained_net(ValData); %[trainingsNum*val_n]. Predicting the outputs 
            end

            pred_y_arr(abs(pred_y_arr-1) > abs(pred_y_arr+1)) = -1; %Outputs nearer to -1 equals -1
            pred_y_arr(abs(pred_y_arr-1) <= abs(pred_y_arr+1)) = 1; %Outputs nearer to 1 equals 1
            

            %Computing accuracy on Validation data points, and then averaging over different trainings:
            val_acc_arr(k) = mean(pred_y_arr ==  ValLabels, [1,2]);
        end
        final_acc = mean(val_acc_arr);
        if final_acc >= best_acc
            best_hiddens = i;
            best_neurons = j;
            best_acc = final_acc;
        end
    end
end

disp("Best accuracy = "+best_acc);
disp("Best number of hidden layers = "+best_hiddens);
disp("Best number of neurons per layer = "+best_neurons);



%Creating the final network:
net = feedforwardnet(repmat(best_neurons, 1, best_hiddens), 'traingd');
net.trainParam.epochs = 500; % Number of training epochs
net.divideParam.trainRatio = 1;
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 0;

view(net);

trained_net = train(net, TrainData, TrainLabels); %Training the network using training data points

%*****************Predicting the outputs using the network:****************
test_pred_y = trained_net(testMat);


test_pred_y(abs(test_pred_y-1) > abs(test_pred_y+1)) = -1; %Outputs nearer to -1 equals -1
test_pred_y(abs(test_pred_y-1) <= abs(test_pred_y+1)) = 1; %Outputs nearer to 1 equals 1

save("Testlabel_MLP.mat", "test_pred_y");
save('best_acc_MLP.mat', 'best_acc');
save('best_hiddens_MLP.mat', 'best_hiddens');
save('best_neurons_MLP.mat', 'best_neurons');
%**************************************************************************



%% Classification: RBF
clc;
close all;

n = size(trainMat, 2); %Total number of training and validation points
val_n = 0.2*n; %Number of validation points


% Defining the range of values to search:
num_neurons_range = 10:5:(n-val_n)/2;
sigma_range = 0.35:0.05:1;

% Optimum values for sigma and the number of neurons:
best_acc = 0;
best_neurons = 0;
best_sigma = 0;

% Loop over the range of values
for n = num_neurons_range
    disp("n = "+n);
    for sigma = sigma_range
        val_acc_arr = zeros(5, 1); %Accuracy values for each training (on validation data points)
        for k = 1:5
            %******************Setting validation points*******************
            valPoints_ind = (k-1)*val_n+1 : k*val_n;
            ValData = trainMat(:, valPoints_ind); %Validation data
            TrainData = trainMat; TrainData(:, valPoints_ind) = []; %Training Data

            ValLabels = Labels(valPoints_ind);
            TrainLabels = Labels; TrainLabels(valPoints_ind) = [];
            %****************************************************************
            

            % Create RBF network:
            net = newrb(TrainData, TrainLabels, 0, sigma, n);
            pred_y_arr = sim(net, ValData); %[1*val_n]. Predicting the outputs 

            pred_y_arr(abs(pred_y_arr-1) > abs(pred_y_arr+1)) = -1; %Outputs nearer to -1 equals -1
            pred_y_arr(abs(pred_y_arr-1) <= abs(pred_y_arr+1)) = 1; %Outputs nearer to 1 equals 1


            %Computing accuracy on Validation data points, and then averaging over different trainings:
            val_acc_arr(k) = mean(pred_y_arr ==  ValLabels);

        end
        final_acc = mean(val_acc_arr);
        % Check if this is the best result so far
        if final_acc >= best_acc
            best_acc = final_acc;
            best_neurons = n;
            best_sigma = sigma;
        end
    end
end

% % Print the best results
disp("Best accuracy: "+ best_acc * 100 + "%");
disp("Best number of neurons: "+ best_neurons);
disp("Best sigma: "+ best_sigma);



%**************Training a network using the best parameters****************
final_net = newrb(TrainData, TrainLabels, 0, best_sigma, best_neurons);
view(final_net);

% Calculate the output labels for the validation data

val_pred_y = sim(final_net, ValData); %NOTE: validation changes with k in the loop. this part is not necessary. Just for checking accuracy again.
test_pred_y = sim(final_net, testMat);

val_pred_y(abs(val_pred_y-1) > abs(val_pred_y+1)) = -1; %Outputs nearer to -1 equals -1
val_pred_y(abs(val_pred_y-1) <= abs(val_pred_y+1)) = 1; %Outputs nearer to 1 equals 1

test_pred_y(abs(test_pred_y-1) > abs(test_pred_y+1)) = -1; %Outputs nearer to -1 equals -1
test_pred_y(abs(test_pred_y-1) <= abs(test_pred_y+1)) = 1; %Outputs nearer to 1 equals 1




save("Testlabel_RBF.mat", "test_pred_y");
save('best_acc_RBF.mat', 'best_acc');
save('best_sigma_RBF.mat', 'best_sigma');
save('best_neurons_RBF.mat', 'best_neurons');
%**************************************************************************












