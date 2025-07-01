%% Metrics
fprintf('Interference calculation step...\n');
T = length(ts);
SINR = NaN(NumGS, T);  % [NumGS x T]
Thrpt = NaN(NumGS, T);  % [NumGS x T]
SE = zeros(1, T);
Noise_mW = 10^(ThermalNoisedBm / 10);
for t = 1:T
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
            Psig_mW = 10^(Psig_dBm / 10);
        elseif isGEOUser
            s_serv = Serv_idxGEOt(userIdx);
            if s_serv == 0 || isnan(s_serv), continue; end
            ch_user = ChannelListGeot(userIdx, s_serv);
            Psig_dBm = PservGEOt(userIdx);
            Psig_mW = 10^(Psig_dBm / 10);
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
                        testInterf_mW = PintLEO_mW + 10^(Pint_dBm/10);
                        SINR_test = Psig_mW / (testInterf_mW + Noise_mW);
                        if 10*log10(SINR_test) < SINRThreshold
                            PintLEO_mW = testInterf_mW;
                            interferersLEO(end+1) = otherIdx;
                        end
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
                        testInterf_mW = PintGEO_mW + 10^(Pint_dBm/10);
                        SINR_test = Psig_mW / (testInterf_mW + Noise_mW);
                        if 10*log10(SINR_test) < SINRThreshold
                            PintGEO_mW = testInterf_mW;
                            interferersGEO(end+1) = otherIdx;
                        end
                    end
                end
            end
        end
        %% Final SINR Computation
        PintTotal_mW = PintLEO_mW + PintGEO_mW;
        Pint_totaldB = 10 * log10(PintTotal_mW + eps);  % avoid log10(0)
        EbN0 = Psig_mW *1e-3 / (Rb * kb * TempK);
        EbN0dB = 10 * log10(EbN0);
        SINR_mW = Psig_mW / (PintTotal_mW + Noise_mW);
        % Thrpt(userIdx, t) = (ChannelBW * log2(1 + SINR_mW));  % Shannon capacity in bits/s
        Thrpt(userIdx, t) = log2(1 + SINR_mW);  % Shannon capacity in bpHz
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
    % Build list of users per channel
    channelUsers = cell(1, numChannels);
    for userIdx = 1:NumGS
        isLEOUser = GSLEOFilter(userIdx);
        isGEOUser = GSGEOFilter(userIdx);
        if isLEOUser
            s_serv = Serv_idxLEOt(userIdx);
            if s_serv == 0 || isnan(s_serv), continue; end
            ch_user = ChannelListLeot(userIdx, s_serv);
        elseif isGEOUser
            s_serv = Serv_idxGEOt(userIdx);
            if s_serv == 0 || isnan(s_serv), continue; end
            ch_user = ChannelListGeot(userIdx, s_serv);
        else
            continue;
        end
        % Only if SINR was computed
        if ~isnan(SINR(userIdx, t))
            channelUsers{ch_user}(end+1) = userIdx;
        end
    end
    % Now count useful vs interfered
    numUseful = 0;
    numInterfered = 0;
    for ch = 1:numChannels
        usersOnCh = channelUsers{ch};
        if isempty(usersOnCh)
            continue;
        end
        % Check if ANY user on this channel is below SINR threshold
        if any(SINR(usersOnCh, t) < SINRThreshold)
            numInterfered = numInterfered + 1;
        else
            numUseful = numUseful + 1;
        end
    end
    % Spectral Efficiency
    SE(t) = numUseful / numChannels;
    fprintf('[t=%d] Useful Channels: %d, Interfered: %d → SE = %.3f\n', ...
        t, numUseful, numInterfered, SE(t));


end
