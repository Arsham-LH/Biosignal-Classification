clear;
% clc;
close all;
load('Project_data.mat');

chanNum = 59;

%Removing Rest Interval
TrainData = TrainData(:, 1000:4000, :);
TestData = TestData(:, 1000:4000, :);

pos_TrainData = TrainData(:, :, TrainLabels == 1);
neg_TrainData = TrainData(:, :, TrainLabels == -1);

neg_labels = size(neg_TrainData,3); %Number of Negative labels
pos_labels = size(pos_TrainData,3); %Number of Positive labels
tot_labels = size(TrainData,3); %Total number of labels

test_tot_labels = size(TestData,3); %Total number of labels in TestData

%% Variance


tot_var_mat = squeeze(var(TrainData, [], 2)); % [channels * 550]. Variance for all channels in train data
test_tot_var_mat = squeeze(var(TestData, [], 2)); % [channels * 159]. For test data. Variance for all channels in test data

%Normalization
tot_var_mat = normalize(tot_var_mat, 2, 'range');
test_tot_var_mat = normalize(test_tot_var_mat, 2, 'range');

%Splitting Variance matrix into Positive and Negative classes
pos_var_mat = tot_var_mat(:, TrainLabels == 1);
neg_var_mat = tot_var_mat(:, TrainLabels == -1);

var_out = zeros(chanNum,1); %[channels*1]

%Computing Fisher criteria
for i = 1:chanNum
    var_out(i) = fisher(pos_var_mat(i,:), neg_var_mat(i,:), tot_var_mat(i,:));
end

disp("Best Fisher for Var: "+max(var_out)); %0.015


%% Amplitude Histigram
%Total amplitude range: from -130 to +130
bins = 50;


pos_ampHist_mat = zeros(chanNum, bins, pos_labels); %[channels*bins*pos_labels]
neg_ampHist_mat = zeros(chanNum, bins, neg_labels); %[channels*bins*neg_labels]
tot_ampHist_mat = zeros(chanNum, bins, tot_labels); %[channels*bins*tot_labels]
test_tot_ampHist_mat = zeros(chanNum, bins, test_tot_labels); %[channels*bins*tot_labels]

ampHist_out = zeros(chanNum,bins); %[channels*bins]

for i = 1:chanNum
    for j = 1:pos_labels
        [pos_N,edges] = histcounts(pos_TrainData(i,:, j), bins);
        pos_ampHist_mat(i,:, j) = pos_N;
    end
end

for i = 1:chanNum
    for j = 1:neg_labels
        [neg_N,edges] = histcounts(neg_TrainData(i,:, j), bins);
        neg_ampHist_mat(i,:, j) = neg_N;
    end
end

for i = 1:chanNum
    for j = 1:tot_labels
        [tot_N,edges] = histcounts(TrainData(i,:, j), bins);
        tot_ampHist_mat(i,:, j) = tot_N;

        if j <= test_tot_labels
            [test_tot_N,edges] = histcounts(TestData(i,:, j), bins);
            test_tot_ampHist_mat(i,:, j) = test_tot_N;
        end
    end
end

%Normalizing features to [0,1]
tot_ampHist_mat = normalize(tot_ampHist_mat, 3, 'range');
test_tot_ampHist_mat = normalize(test_tot_ampHist_mat, 3, 'range');

pos_ampHist_mat = tot_ampHist_mat(:, :, TrainLabels == 1);
neg_ampHist_mat = tot_ampHist_mat(:, :, TrainLabels == -1);

for ch = 1:chanNum
    for bin = 1:bins
        ampHist_out(ch,bin) = fisher(pos_ampHist_mat(ch,bin,:), ...
            neg_ampHist_mat(ch,bin,:), tot_ampHist_mat(ch,bin,:));
    end
end

disp("Best Fisher for AmpHist with Bins "+bins+": "+max(max(ampHist_out))); %0.026



%% AR Model
order = 10;


pos_AR_mat = zeros(chanNum, order, pos_labels); %[channels*order*pos_labels]
neg_AR_mat = zeros(chanNum, order, neg_labels); %[channels*order*neg_labels]
tot_AR_mat = zeros(chanNum, order, tot_labels); %[channels*order*tot_labels]
test_tot_AR_mat = zeros(chanNum, order, test_tot_labels); %[channels*order*tot_labels]

AR_out = zeros(chanNum,order); %[channels*order]

for i = 1:chanNum
    for j = 1:pos_labels
%         pos_sys = ar(pos_TrainData(i,:, j), order, 'Ts', 1/fs);
%         pos_AR_mat(i,:, j) = pos_sys.A(2:end); %Ignoring coeff 1 at the first index

        pos_sys = aryule(pos_TrainData(i,:, j), order);
        pos_AR_mat(i,:, j) = pos_sys(2:end); %Ignoring coeff 1 at the first index

    end
end

for i = 1:chanNum
    for j = 1:neg_labels
%         neg_sys = ar(neg_TrainData(i,:, j), order, 'Ts', 1/fs);
%         neg_AR_mat(i,:, j) = neg_sys.A(2:end); %Ignoring coeff 1 at the first index

        neg_sys = aryule(neg_TrainData(i,:, j), order);
        neg_AR_mat(i,:, j) = neg_sys(2:end); %Ignoring coeff 1 at the first index
    end
end

for i = 1:chanNum
    for j = 1:tot_labels
        tot_sys = aryule(TrainData(i,:, j), order);
        tot_AR_mat(i,:, j) = tot_sys(2:end); %Ignoring coeff 1 at the first index

        if j <= test_tot_labels
            test_tot_sys = aryule(TestData(i,:, j), order);
            test_tot_AR_mat(i,:, j) = test_tot_sys(2:end); %Ignoring coeff 1 at the first index
        end
    end
end
tot_AR_mat = normalize(tot_AR_mat, 3, 'range');
test_tot_AR_mat = normalize(test_tot_AR_mat, 3, 'range');

pos_AR_mat = tot_AR_mat(:, :, TrainLabels == 1);
neg_AR_mat = tot_AR_mat(:, :, TrainLabels == -1);


for ch = 1:chanNum
    for o = 1:order
        AR_out(ch,o) = fisher(pos_AR_mat(ch,o,:), ...
            neg_AR_mat(ch,o,:), tot_AR_mat(ch,o,:));
    end
end

disp("Best Fisher for AR with Order "+order+": "+max(max(AR_out))); %0.013

%% Form Factor

pos_TrainData_diff = diff(pos_TrainData, 1, 2);
neg_TrainData_diff = diff(neg_TrainData, 1, 2);
TrainData_diff = diff(TrainData, 1, 2);
TestData_diff = diff(TestData, 1, 2);

pos_TrainData_diff2 = diff(pos_TrainData, 2, 2);
neg_TrainData_diff2 = diff(neg_TrainData, 2, 2);
TrainData_diff2 = diff(TrainData, 2, 2);
TestData_diff2 = diff(TestData, 2, 2);


pos_FF_mat = (var(pos_TrainData_diff2,[],2)./var(pos_TrainData_diff2,[],2)) ./ (var(pos_TrainData_diff,[],2)./var(pos_TrainData,[],2)); %[channels * posLabels]
neg_FF_mat = (var(neg_TrainData_diff2,[],2)./var(neg_TrainData_diff2,[],2)) ./ (var(neg_TrainData_diff,[],2)./var(neg_TrainData,[],2)); %[channels * negLabels]
tot_FF_mat = squeeze((var(TrainData_diff2,[],2)./var(TrainData_diff2,[],2)) ./ (var(TrainData_diff,[],2)./var(TrainData,[],2))); %[channels * totLabels]
test_tot_FF_mat = squeeze((var(TestData_diff2,[],2)./var(TestData_diff2,[],2)) ./ (var(TestData_diff,[],2)./var(TestData,[],2))); %[channels * test_totLabels]

tot_FF_mat = normalize(tot_FF_mat, 2, 'range');
test_tot_FF_mat = normalize(test_tot_FF_mat, 2, 'range');

pos_FF_mat = tot_FF_mat(:, TrainLabels == 1);
neg_FF_mat = tot_FF_mat(:, TrainLabels == -1);

FF_out = zeros(chanNum,1); %[channels*1]

for i = 1:chanNum
    FF_out(i) = fisher(pos_FF_mat(i,:), neg_FF_mat(i,:), tot_FF_mat(i,:));
end

disp("Best Fisher for FF: "+max(FF_out)); %0.026

%% Correlation


pos_corr_mat = zeros(chanNum, chanNum, pos_labels); %[chans*chans*pos_labels]
neg_corr_mat = zeros(chanNum, chanNum, neg_labels); %[chans*chans*neg_labels]
tot_corr_mat = zeros(chanNum, chanNum, tot_labels); %[chans*chans*tot_labels]
test_tot_corr_mat = zeros(chanNum, chanNum, test_tot_labels); %[chans*chans*test_tot_labels]

corr_out = zeros(chanNum,chanNum); %[chans*chans]

for i = 1:pos_labels
    pos_corr_mat(:,:, i) = corr(squeeze(pos_TrainData(:,:,i)).');
end

for i = 1:neg_labels
    neg_corr_mat(:,:, i) = corr(squeeze(neg_TrainData(:,:,i)).');
end

for i = 1:tot_labels
    tot_corr_mat(:,:, i) = corr(squeeze(TrainData(:,:,i)).');
    if i <= test_tot_labels
        test_tot_corr_mat(:,:, i) = corr(squeeze(TestData(:,:,i)).');
    end
end
tot_corr_mat = normalize(tot_corr_mat, 3, 'range');
test_tot_corr_mat = normalize(test_tot_corr_mat, 3, 'range');

pos_corr_mat = tot_corr_mat(:, :, TrainLabels == 1);
neg_corr_mat = tot_corr_mat(:, :, TrainLabels == -1);


for ch1 = 1:chanNum
    for ch2 = 1:chanNum
        corr_out(ch1,ch2) = fisher(pos_corr_mat(ch1,ch2,:), ...
            neg_corr_mat(ch1,ch2,:), tot_corr_mat(ch1,ch2,:));
    end
end

corr_out = triu(corr_out, 1);
disp("Best Fisher for Pearson corr: "+max(max(corr_out))); %0.026

%% Integrating Statistical Features

N_var = length(var_out);
N_ampHist = length(ampHist_out(:));
N_AR = length(AR_out(:));
N_FF = length(FF_out);
N_corr = length(corr_out(:));

N_arr_stat = [N_var; N_ampHist; N_AR; N_FF; N_corr];
statFeat_startIDX = zeros(length(N_arr_stat),1); %statistical features start index

for i = 1:length(N_arr_stat)
    statFeat_startIDX(i) = sum(N_arr_stat(1:i));
end


stat_features = [var_out; ampHist_out(:); AR_out(:); FF_out; corr_out(:)];
statFeaturesNum = 30; %Number of statistical features to keep

%Keeping best statistical features:
[slct_stat_features, slct_stat_idx] = maxk(stat_features, statFeaturesNum);

[slct_stat_idx, I] = sort(slct_stat_idx);
slct_stat_features = slct_stat_features(I);


slct_var_chans = []; %list of selected channels for var
slct_ampHist_chans = []; %list of selected channels for ampHist
slct_ampHist_bins = []; %list of selected bins for ampHist
slct_AR_chans = []; %list of selected channels for AR
slct_AR_params = []; %list of selected parameters for AR
slct_FF_chans = []; %list of selected channels for FF
slct_corr_chan1 = []; %list of selected channels 1 for correlation
slct_corr_chan2 = []; %list of selected channels 2 for correlation




for i = 1:statFeaturesNum
    idx = slct_stat_idx(i);
    if idx <= statFeat_startIDX(1)
        slct_var_chans(end+1) = idx;

    elseif idx > statFeat_startIDX(1) && idx <= statFeat_startIDX(2)
        tmp_idx = idx - statFeat_startIDX(1);
        slct_ampHist_chans(end+1) = mod(tmp_idx - 1, chanNum) + 1; %Row number in ampHist_out
        slct_ampHist_bins(end+1) = ceil(tmp_idx / chanNum); %col number in ampHist_out

    elseif idx > statFeat_startIDX(2) && idx <= statFeat_startIDX(3)
        tmp_idx = idx - statFeat_startIDX(2);
        slct_AR_chans(end+1) = mod(tmp_idx - 1, chanNum) + 1; %Row number in AR_out
        slct_AR_params(end+1) = ceil(tmp_idx / chanNum); %col number in AR_out

    elseif idx > statFeat_startIDX(3) && idx <= statFeat_startIDX(4)
        slct_FF_chans(end+1) = idx - statFeat_startIDX(3);

    else
        tmp_idx = idx - statFeat_startIDX(4);
        slct_corr_chan1(end+1) = mod(tmp_idx - 1, chanNum) + 1; %Row number in corr_out
        slct_corr_chan2(end+1) = ceil(tmp_idx / chanNum); %col number in corr_out
    end
end

%% Maxfreq


pos_maxfreq_mat = zeros(chanNum, pos_labels); % [channels * posLabels]
neg_maxfreq_mat = zeros(chanNum, neg_labels); % [channels * negLabels]
tot_maxfreq_mat = zeros(chanNum, tot_labels); % [channels * totLabels]
test_tot_maxfreq_mat = zeros(chanNum, test_tot_labels); % [channels * test_totLabels]

maxfreq_out = zeros(chanNum,1); %[channels*1]

for i = 1:chanNum
    %Computing Spectrum
    [pos_psd_mat, pos_f] = pwelch(squeeze(pos_TrainData(i,:,:)), [], [], [], fs);
    [neg_psd_mat, neg_f] = pwelch(squeeze(neg_TrainData(i,:,:)), [], [], [], fs);
    [tot_psd_mat, tot_f] = pwelch(squeeze(TrainData(i,:,:)), [], [], [], fs); 
    [test_tot_psd_mat, test_tot_f] = pwelch(squeeze(TestData(i,:,:)), [], [], [], fs); 

    %Finding Frequency with maximum power in the spectrum:
    for j = 1:pos_labels
        pos_maxfreq_mat(i,j) = pos_f(pos_psd_mat(:, j) == max(pos_psd_mat(:, j)));
    end

    for j = 1:neg_labels
        neg_maxfreq_mat(i,j) = neg_f(neg_psd_mat(:, j) == max(neg_psd_mat(:, j)));
    end

    for j = 1:tot_labels
        tot_maxfreq_mat(i,j) = tot_f(tot_psd_mat(:, j) == max(tot_psd_mat(:, j)));
        if j <= test_tot_labels
            test_tot_maxfreq_mat(i,j) = test_tot_f(test_tot_psd_mat(:, j) == max(test_tot_psd_mat(:, j)));
        end
    end

end

tot_maxfreq_mat = normalize(tot_maxfreq_mat, 2, 'range');
test_tot_maxfreq_mat = normalize(test_tot_maxfreq_mat, 2, 'range');

pos_maxfreq_mat = tot_maxfreq_mat(:, TrainLabels == 1);
neg_maxfreq_mat = tot_maxfreq_mat(:, TrainLabels == -1);

for i = 1:chanNum
    maxfreq_out(i) = fisher(pos_maxfreq_mat(i,:), neg_maxfreq_mat(i,:), tot_maxfreq_mat(i,:));
end


disp("Best Fisher for Maxfreq: "+max(maxfreq_out)); %0.020



%% Meanfreq

pos_meanfreq_mat = zeros(chanNum, pos_labels); % [channels * posLabels]
neg_meanfreq_mat = zeros(chanNum, neg_labels); % [channels * negLabels]
tot_meanfreq_mat = zeros(chanNum, tot_labels); % [channels * totLabels]
test_tot_meanfreq_mat = zeros(chanNum, test_tot_labels); % [channels * test_totLabels]

meanfreq_out = zeros(chanNum,1); %[channels*1]

for i = 1:chanNum
    pos_meanfreq_mat(i,:) = meanfreq(squeeze(pos_TrainData(i,:,:)), fs);
    neg_meanfreq_mat(i,:) = meanfreq(squeeze(neg_TrainData(i,:,:)), fs);
    tot_meanfreq_mat(i,:) = meanfreq(squeeze(TrainData(i,:,:)), fs);
    test_tot_meanfreq_mat(i,:) = meanfreq(squeeze(TestData(i,:,:)), fs);

    tot_meanfreq_mat(i,:) = normalize(tot_meanfreq_mat(i,:),2, 'range');
    test_tot_meanfreq_mat(i,:) = normalize(test_tot_meanfreq_mat(i,:),2, 'range');

    pos_meanfreq_mat(i,:) = tot_meanfreq_mat(i, TrainLabels == 1);
    neg_meanfreq_mat(i,:) = tot_meanfreq_mat(i, TrainLabels == -1);

    meanfreq_out(i) = fisher(pos_meanfreq_mat(i,:), neg_meanfreq_mat(i,:), tot_meanfreq_mat(i,:));
end


disp("Best Fisher for Meanfreq: "+max(meanfreq_out)); %0.025



%% Medianfreq

pos_medfreq_mat = zeros(chanNum, pos_labels); % [channels * posLabels]
neg_medfreq_mat = zeros(chanNum, neg_labels); % [channels * negLabels]
tot_medfreq_mat = zeros(chanNum, tot_labels); % [channels * totLabels]
test_tot_medfreq_mat = zeros(chanNum, test_tot_labels); % [channels * test_totLabels]

medfreq_out = zeros(chanNum,1); %[channels*1]

for i = 1:chanNum
    pos_medfreq_mat(i,:) = medfreq(squeeze(pos_TrainData(i,:,:)), fs);
    neg_medfreq_mat(i,:) = medfreq(squeeze(neg_TrainData(i,:,:)), fs);
    tot_medfreq_mat(i,:) = medfreq(squeeze(TrainData(i,:,:)), fs);
    test_tot_medfreq_mat(i,:) = medfreq(squeeze(TestData(i,:,:)), fs);
    
    tot_medfreq_mat(i,:) = normalize(tot_medfreq_mat(i,:),2, 'range');
    test_tot_medfreq_mat(i,:) = normalize(test_tot_medfreq_mat(i,:),2, 'range');

    pos_medfreq_mat(i,:) = tot_medfreq_mat(i, TrainLabels == 1);
    neg_medfreq_mat(i,:) = tot_medfreq_mat(i, TrainLabels == -1);
    
    medfreq_out(i) = fisher(pos_medfreq_mat(i,:), neg_medfreq_mat(i,:), tot_medfreq_mat(i,:));
end


disp("Best Fisher for Medfreq: "+max(medfreq_out)); %0.027


%% Bandpower

pos_bp_mat = zeros(chanNum, 5, pos_labels); % [channels * freqRange * posLabels]
neg_bp_mat = zeros(chanNum, 5, neg_labels); % [channels * freqRange * negLabels]
tot_bp_mat = zeros(chanNum, 5, tot_labels); % [channels * freqRange * totLabels]
test_tot_bp_mat = zeros(chanNum, 5, test_tot_labels); % [channels * * freqRange test_totLabels]

bp_out = zeros(chanNum,5); %[channels * freqRange]

for i = 1:chanNum
    for fr = 1:5
        switch(fr)
            case 1
                frange = [0.1 3]; %Delta
            case 2
                frange = [4 7]; %Theta
            case 3
                frange = [8 12]; %Alpha
            case 4
                frange = [12 30]; %Beta
            case 5
                frange = [30 100]; %Gamma
        end

        pos_bp_mat(i,fr, :) = bandpower(squeeze(pos_TrainData(i,:,:)), fs, frange);
        neg_bp_mat(i,fr, :) = bandpower(squeeze(neg_TrainData(i,:,:)), fs, frange);
        tot_bp_mat(i,fr, :) = bandpower(squeeze(TrainData(i,:,:)), fs, frange);
        test_tot_bp_mat(i,fr, :) = bandpower(squeeze(TestData(i,:,:)), fs, frange);

        tot_bp_mat(i,fr,:) = normalize(squeeze(tot_bp_mat(i,fr,:)), 'range');
        test_tot_bp_mat(i,fr,:) = normalize(squeeze(test_tot_bp_mat(i,fr,:)), 'range');

        pos_bp_mat(i,fr,:) = tot_bp_mat(i,fr, TrainLabels == 1);
        neg_bp_mat(i,fr,:) = tot_bp_mat(i,fr, TrainLabels == -1);

        bp_out(i, fr) = fisher(pos_bp_mat(i,fr,:), neg_bp_mat(i,fr,:), tot_bp_mat(i,fr,:));
    end
end



disp("Best Fisher for Bandpower: "+max(max(bp_out))); %0.063



%% Integrating Frequency Features

N_maxfreq = length(maxfreq_out);
N_meanfreq = length(meanfreq_out);
N_medfreq = length(medfreq_out);
N_bp = length(bp_out(:));

N_arr_freq = [N_maxfreq; N_meanfreq; N_medfreq; N_bp];
freqFeat_startIDX = zeros(length(N_arr_freq),1); %frequenxy features start index
for i = 1:length(N_arr_freq)
    freqFeat_startIDX(i) = sum(N_arr_freq(1:i));
end

slct_maxfreq_chans = []; %list of selected channels for maxfreq
slct_meanfreq_chans = []; %list of selected channels for meanfreq
slct_medfreq_chans = []; %list of selected channels for medfreq
slct_bp_chans = []; %list of selected channels for bandpower
slct_bp_bands = []; %list of selected bands for bandpower

freq_features = [maxfreq_out; meanfreq_out; medfreq_out; bp_out(:)];

freqFeaturesNum = 30; %Number of frequency features to keep

[slct_freq_features, slct_freq_idx] = maxk(freq_features, freqFeaturesNum);

[slct_freq_idx, I] = sort(slct_freq_idx);
slct_freq_features = slct_freq_features(I);

for i = 1:freqFeaturesNum
    idx = slct_freq_idx(i);
    if idx <= freqFeat_startIDX(1)
        slct_maxfreq_chans(end+1) = idx;

    elseif idx > freqFeat_startIDX(1) && idx <= freqFeat_startIDX(2)
        slct_meanfreq_chans(end+1) = idx - freqFeat_startIDX(1);

    elseif idx > freqFeat_startIDX(2) && idx <= freqFeat_startIDX(3)
        slct_medfreq_chans(end+1) = idx - freqFeat_startIDX(2);

    else
        bp_idx = idx - freqFeat_startIDX(3);
        slct_bp_chans(end+1) = mod(bp_idx - 1, chanNum) + 1; %Row number in bp_out
        slct_bp_bands(end+1) = ceil(bp_idx / chanNum); %col number ainMatin bp_out
    end
end



%% Creating TrainMat

%*******************Adding statistical features********************
trainMat = tot_var_mat(slct_var_chans,:);
testMat = test_tot_var_mat(slct_var_chans,:);

for i = 1:length(slct_ampHist_chans)
    trainMat = [trainMat; squeeze(tot_ampHist_mat(slct_ampHist_chans(i), slct_ampHist_bins(i),:)).'];
    testMat = [testMat; squeeze(test_tot_ampHist_mat(slct_ampHist_chans(i), slct_ampHist_bins(i),:)).'];
end

for i = 1:length(slct_AR_chans)
    trainMat = [trainMat; squeeze(tot_AR_mat(slct_AR_chans(i), slct_AR_params(i),:)).'];
    testMat = [testMat; squeeze(test_tot_AR_mat(slct_AR_chans(i), slct_AR_params(i),:)).'];
end

trainMat = [trainMat; tot_FF_mat(slct_FF_chans, :)];
testMat = [testMat; test_tot_FF_mat(slct_FF_chans, :)];

for i = 1:length(slct_corr_chan1)
    trainMat = [trainMat; squeeze(tot_corr_mat(slct_corr_chan1(i), slct_corr_chan2(i),:)).'];
    testMat = [testMat; squeeze(test_tot_corr_mat(slct_corr_chan1(i), slct_corr_chan2(i),:)).'];
end
%****************************************************************


%*******************Adding frequency features********************
trainMat = [trainMat; tot_maxfreq_mat(slct_maxfreq_chans, :)];
testMat = [testMat; test_tot_maxfreq_mat(slct_maxfreq_chans, :)];

trainMat = [trainMat; tot_meanfreq_mat(slct_meanfreq_chans, :)];
testMat = [testMat; test_tot_meanfreq_mat(slct_meanfreq_chans, :)];

trainMat = [trainMat; tot_medfreq_mat(slct_medfreq_chans, :)];
testMat = [testMat; test_tot_medfreq_mat(slct_medfreq_chans, :)];

for i = 1:length(slct_bp_chans)
    trainMat = [trainMat; squeeze(tot_bp_mat(slct_bp_chans(i), slct_bp_bands(i), :)).'];
    testMat = [testMat; squeeze(test_tot_bp_mat(slct_bp_chans(i), slct_bp_bands(i), :)).'];
end
%****************************************************************


save("trainMat.mat", 'trainMat');
save("testMat.mat", 'testMat');
save("TrainLabels.mat", 'TrainLabels');

%% functions

function out = fisher(pos_feats, neg_feats, tot_feats)
% [pos_feats] = posLabels*1
% [neg_feats] = negLabels*1
% [tot_feats] = 550*1

    u1 = mean(pos_feats);
    u2 = mean(neg_feats);
    u0 = mean(tot_feats);

    var1 = var(pos_feats);
    var2 = var(neg_feats);

    out = (abs(u1-u0)^2 + abs(u2-u0)^2) / (var1+var2);
end



