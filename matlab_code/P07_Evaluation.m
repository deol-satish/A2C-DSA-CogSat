%%
fprintf('Interference calculation step...\n');
T = length(ts);
SINR = NaN(NumGS, T);  % [NumGS x T]
SE = zeros(1, T);

for t = 1:T
    % PrxLEOt = ActualPrxLEO(:, :, t);      % [NumGS x LEO]
    % PrxGEOt = ActualPrxGEO(:, :, t);
    PrxLEOt = PrxLEO(:, :, t);              % [NumGS x LEO]
    PrxGEOt = PrxGEO(:, :, t);              % [NumGS x GEO]
    ChannelListLeot = ChannelListLeo(:, :, t);
    ChannelListGeot = ChannelListGeo(:, :, t);
    PservLEOt = PservLEO(:, t);
    Serv_idxLEOt = Serv_idxLEO(:, t);
    PservGEOt = PservGEO(:, t);
    Serv_idxGEOt = Serv_idxGEO(:, t);
    for userIdx = 1:NumGS
        isLEOUser = GSLEOFilter(userIdx);
        isGEOUser = GSGEOFilter(userIdx);
        if isLEOUser
            s_serv = Serv_idxLEOt(userIdx);
            if s_serv == 0 || isnan(s_serv), continue; end
            ch_user = ChannelListLeot(userIdx, s_serv);
            Psig_dBm = PservLEOt(userIdx);
        elseif isGEOUser
            s_serv = Serv_idxGEOt(userIdx);
            if s_serv == 0 || isnan(s_serv), continue; end
            ch_user = ChannelListGeot(userIdx, s_serv);
            Psig_dBm = PservGEOt(userIdx);
        else
            continue;  % undefined user
        end
        %% Interference from LEO
        PintLEO_mW = 0;
        interferersLEO = [];  % <=== store interfering user indices
        for s = 1:leoNum
            for otherIdx = 1:NumGS
                if otherIdx == userIdx || GSLEOFilter(otherIdx) == 0, continue; end
                ch_other = ChannelListLeot(otherIdx, s);
                if ch_other == ch_user
                    Pint_dBm = PrxLEOt(userIdx, s);
                    if ~isnan(Pint_dBm) && ~isinf(Pint_dBm)
                        PintLEO_mW = PintLEO_mW + 10^(Pint_dBm / 10);
                        interferersLEO(end+1) = otherIdx;  
                    end
                end
            end
        end
        %% Interference from GEO
        PintGEO_mW = 0;
        interferersGEO = [];  % <=== store interfering user indices
        for g = 1:geoNum
            for otherIdx = 1:NumGS
                if otherIdx == userIdx || GSGEOFilter(otherIdx) == 0, continue; end
                ch_other = ChannelListGeot(otherIdx, g);
                if ch_other == ch_user
                    Pint_dBm = PrxGEOt(userIdx, g);
                    if ~isnan(Pint_dBm) && ~isinf(Pint_dBm)
                        PintGEO_mW = PintGEO_mW + 10^(Pint_dBm / 10);
                        interferersGEO(end+1) = otherIdx;  % <=== save
                    end
                end
            end
        end
        %% Final SINR Computation
        PintTotal_mW = PintLEO_mW + PintGEO_mW;
        Pint_totaldB = 10 * log10(PintTotal_mW + eps);  % avoid log10(0)
        Psig_mW = 10^(Psig_dBm / 10);
        Noise_mW = 10^(ThermalNoisedBm / 10);
        EbN0 = Psig_mW *1e-3 / (Rb * kb * TempK);
        EbN0dB = 10 * log10(EbN0);
        SINR_mW = Psig_mW / (PintTotal_mW + Noise_mW);
        % Thrpt(userIdx, t) = (ChannelBW * log2(1 + SINR_mW));  % Shannon capacity in mbits/s
        Thrpt(userIdx, t) = log2(1 + SINR_mW);  % Shannon capacity in mbits/s
        SINR(userIdx, t) = 10 * log10(SINR_mW);
        %% Print full debug info
        fprintf('[t=%d] User %d → Channel %d: Psig=%.2f dBm, Interf=%.2f dBm, SINR=%.2f dB\n', ...
            t, userIdx, ch_user, Psig_dBm, Pint_totaldB, SINR(userIdx, t));
        
        if ~isempty(interferersLEO)
            fprintf('    ↳ LEO Interferers: %s\n', mat2str(interferersLEO));
        end
        if ~isempty(interferersGEO)
            fprintf('    ↳ GEO Interferers: %s\n', mat2str(interferersGEO));
        end
    end
    % Initialize containers for channels actively used at this time step
    activeChannels = [];    % Active links with service
    channelUsage = zeros(1, numChannels);  % Channel usage counter
    
    % Loop over users and count actual, non-interfered usage
    for userIdx = 1:NumGS
        isLEOUser = GSLEOFilter(userIdx);
        isGEOUser = GSGEOFilter(userIdx);
    
        if isLEOUser
            s_serv = Serv_idxLEOt(userIdx);
            if s_serv == 0 || isnan(s_serv), continue; end
            ch_user = ChannelListLeot(userIdx, s_serv);
            Prx_dBm = PrxLEOt(userIdx, s_serv);
        elseif isGEOUser
            s_serv = Serv_idxGEOt(userIdx);
            if s_serv == 0 || isnan(s_serv), continue; end
            ch_user = ChannelListGeot(userIdx, s_serv);
            Prx_dBm = PrxGEOt(userIdx, s_serv);
        else
            continue;
        end
    
        % Only count channels with active links above noise floor
        if ~isnan(Prx_dBm) && Prx_dBm > ThermalNoisedBm
            activeChannels(end+1) = ch_user;
            channelUsage(ch_user) = channelUsage(ch_user) + 1;
        end
    end
    
    % Determine how many channels are reused (i.e., interfered)
    numUsed = nnz(channelUsage > 0);
    numInterfered = nnz(channelUsage > 1);
    
    % Spectral Efficiency
    SE(t) = (numUsed - numInterfered) / numChannels;
    SE(t) = max(0, min(1, SE(t)));  % clip to [0,1] if needed
    
    fprintf('[t=%d] Active Channels: %d, Interfered: %d → SE = %.3f\n', ...
            t, numUsed, numInterfered, SE(t));

end
