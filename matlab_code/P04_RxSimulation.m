%% P04_RxSimulation
%% Gain Calculation
%% Rx Gain
Grx = 10* log10((pi * gsAntenna *fc /c)^2 * eff);
ThermalNoisedBm = 10 * log10(kb * TempK * ChannelBW) +30; % Noise in dBm
%% LEO Tx Gain
%% Define sat gain based on the antenna length - parabolic antenna case (1)
% GtxLEO = 10* log10((pi * leo.Antenna *fc /c)^2 * eff);
%% Define a realistic 1D sinc-squared approximation pattern shape antenna case (2)
GtxLEO = leo.GainMax + 10 * log10( (sinc(1.391 * leotheta / leo.psi)).^2 );
% Gain Pattern: Cosine Power Model with Azimuth Ripple
% figure; plot(squeeze(leotheta(1,1,:)), squeeze(GtxLEO(1,1,:)), 'LineWidth', 2);
%% 2D Sinc^2 Antenna Gain Pattern => Beamwidth control case (3)
% leo.AntShape = 0.573 * leo.psi + 0.1;  % leo.psi in radians
% GtxLEO = leo.GainMax + 10 * log10(( abs(sinc(leo.AntShape*leotheta / leo.psi)).^2 ) ...
%            .* ( abs(  sinc(leo.AntShape*leoAzimuth / leo.psi)).^2 ));
% figure; plot(squeeze(leotheta(1,1,:)), squeeze(GtxLEO(1,1,:)), 'LineWidth', 2);
%% GEO Tx Gain
GtxGEO = 10* log10((pi * geoAntenna *fc /c)^2 * eff);
%% LEO Power calculations
RhoLEO(ElLEO<0) = Inf;
PathLoss = 20*log10(fc) + 20*log10(RhoLEO) -147.55;
AtmoLLEO = F01_ComputeAtmosphericLoss(fc, ElLEO, Att);
FadingLEO = F02_MultipathFadingLoss(FadingModel, ElLEO);
PrxLEO = leoPower + GtxLEO + Grx - PathLoss - AtmoLLEO - FadingLEO;
SNRLEO = PrxLEO - ThermalNoisedBm;
%% GEO Power calculations
RhoGEO(ElGEO<0) = Inf;
PathLoss = 20*log10(fc) + 20*log10(RhoGEO) -147.55;
AtmoLGEO = F01_ComputeAtmosphericLoss(fc, ElGEO , Att);
FadingGEO = F02_MultipathFadingLoss(FadingModel, ElGEO);
PrxGEO = geoPower + GtxGEO + Grx - PathLoss - AtmoLGEO - FadingGEO;
SNRGEO = PrxGEO - ThermalNoisedBm;
