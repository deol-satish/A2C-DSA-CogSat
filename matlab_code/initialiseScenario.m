%% Main Script to define the Geometrical simulator, Receivers, and Interference
clear; clc;close all hidden
%% Define Parameters and Ground stations
P01_Parameters
P02_GStations
%% Create Scenario
fprintf('Creating satellite scenario...\n');
sc = satelliteScenario(startTime, stopTime, sampleTime);
%% Satellite and GS creation
P03_GeometricSimulatoion
% play(sc,PlaybackSpeedMultiplier=100);
% save('Geometric',"gsAntenna","fc","c","leoAntenna","ElLEO","RhoLEO","eff", ...
%     "geoAntenna","ElGEO", "RhoGEO", "NumLeoUser", "NumGeoUser", "ts", "GS", ...
%     "leoNum", "geoNum", "GSLEOFilter", "GSGEOFilter", "leoPower", "geoPower");
%% Power Simulation
% clear; clc;close all hidden
% load("Geometric")
P04_RxSimulation
% save('Power',"PrxLEO","PrxGEO");
%% Channel allocation and Serving satellites
% clear; clc;close all hidden
% load("Geometric")
% load('Power');
P05_ChannelAllocation
% save('Data');


%% Interference calculation Next
T = length(ts);
SINR = NaN(NumGS, T);  % [NumGS x T]
Intf = NaN(NumGS, T);  % [NumGS x T]
Thrpt = NaN(NumGS, T);  % [NumGS x T]
SINR_mW_dict = NaN(NumGS, T);  % [NumGS x T]
Intf_mW_dict = NaN(NumGS, T);  % [NumGS x T]
Noise_mW = 10^(ThermalNoisedBm / 10);
SE = zeros(1, T);

FreqAlloc = NaN(NumGS, length(ts));  % Initialize
t = 1