% BME project: wrist movement classification
% EMG preprocessing

close all
% Coneverting .adicht to .mat format
% adi.convert("Exp1.adicht");

load("Exp1.mat");
% Class labels
label_names = ["palm", "wrist up-down", "wrist rotate", "palm up-down"];

% Start and end of movements
periods = [[5, 13.5]; [1, 11]; [0.5, 14.5]; [1.5, 13.5]];

% Sampling frequency
fs = 200;

channel1_filtered = zeros(4, 3500);
channel2_filtered = zeros(4, 3500);

fcutlow=1;   %Low cut frequency in Hz

% Creating butterworth filter 
[b,a] = butter(1, fcutlow/(fs/2), "high");
[b50,a50] = butter(1, [49.5 50.5]/(fs/2), "stop");

% Filtering signals

for i=1:4
    channel1 = eval("data__chan_1_rec_"+num2str(i+2));
    channel2 = eval("data__chan_2_rec_"+num2str(i+2));
    channel1_filtered(i, 1:length(channel1)) = filter(b, a, channel1);
    channel2_filtered(i, 1:length(channel2)) = filter(b, a, channel2);
    channel1_filtered(i, 1:length(channel1)) = filter(b50, a50, channel1_filtered(i, 1:length(channel1)));
    channel2_filtered(i, 1:length(channel2)) = filter(b50, a50, channel2_filtered(i, 1:length(channel2)));
    
    % Plotting spectrograms before and after filtering
    figure
    subplot(4, 1, 1)
    pspectrum(channel1,fs,'spectrogram','FrequencyLimits',[0 100])
    title("channel 1 before filtering")
    subplot(4, 1, 2)
    pspectrum(channel1_filtered(i, 1:length(channel1)),fs,'spectrogram','FrequencyLimits',[0 100])
    title("channel 1 after filtering")
    subplot(4, 1, 3)   
    pspectrum(channel2,fs,'spectrogram','FrequencyLimits',[0 100])
        title("channel 2 before filtering")
    subplot(4, 1, 4)
    pspectrum(channel2_filtered(i, 1:length(channel1)),fs,'spectrogram','FrequencyLimits',[0 100])
    title("channel 2 after filtering")

    sgtitle(label_names(i) + " spectrogram")
    saveas(gcf, label_names(i) + " spectrogram.png")
end


% Plotting power spectral density of signals
figure
[Pxx, F] = pwelch(channel1_filtered(1, :), [], [], [], fs);
subplot(2, 1, 1)
plot(F, Pxx)
title("Channel 1")
xlabel("Frequency(Hz)")

[Pxx, F] = pwelch(channel2_filtered(1, :), [], [], [], fs);
subplot(2, 1, 2)
plot(F, Pxx)
title("Channel 2")
xlabel("Frequency(Hz)")

sgtitle("PSD of EMG channels")
saveas(gcf, "emg psd.png")


% Normalizing signals (Applying z-score)
signal1 = normalize(channel1_filtered')';
signal2 = normalize(channel2_filtered')';


% Cropping baselines
signal1_cropped = {};
signal2_cropped = {};

for i=1:4
    signal1_cropped{end+1} = signal1(i, periods(i, 1)*fs: periods(i, 2)*fs);
    signal2_cropped{end+1} = signal2(i, periods(i, 1)*fs: periods(i, 2)*fs);
end


% Windowing signals
W = 0.2*fs;
overlap = 0.19*fs;
data = []; % [totalWindows*channels*windowLen]
y = []; %Labels

start_index = [];
for i=1:4
    start_index(end+1) = length(y)+1; %Number of the first index whose label is i (ignore the first element)
    L = length(signal1_cropped{i});
    n_windows = (L-overlap)/(W-overlap);
    for j=1:n_windows
        data(end+1, 1, :) = signal1_cropped{i}(1+(j-1)*(W-overlap): (j-1)*(W-overlap)+W);
        data(end, 2, :) = signal2_cropped{i}(1+(j-1)*(W-overlap): (j-1)*(W-overlap)+W);
        y(end+1) = i;
    end
end





