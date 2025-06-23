 %% Visualization

Scale = 0.7;
h_Fig=figure('PaperPositionMode', 'manual','PaperUnits','inches','PaperPosition',[0 0 3.5*2 3.5*2/1.618*Scale],'Position',[200 300 800 800/1.618*Scale]);
histogram(SINR(:), 'BinWidth', 0.5, 'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'k');
xlabel('SINR [dB]');
ylabel('Frequency');
title('SINR Distribution Across All Users and Time Steps');
grid on;
xlim([min(SINR(:))-1, max(SINR(:))+1]);
ax=gca;
grid on;ax.Box = 'on';set(gca,'LooseInset',get(gca,'TightInset'),'FontSize',12);
ax.LineWidth = 1;  % Set box (axis) thickness to 1.5 points
saveLocation ="./";
Filename = fullfile(saveLocation, 'Figure1');
print(h_Fig, '-dpng','-r600',Filename)
%% Compute average SINR per user
meanSINR = mean(SINR, 2, 'omitnan');
Scale = 0.7;
h_Fig=figure('PaperPositionMode', 'manual','PaperUnits','inches','PaperPosition',[0 0 3.5*2 3.5*2/1.618*Scale],'Position',[200 300 800 800/1.618*Scale]);
plot(1:NumGS, meanSINR,'k-o','LineWidth',1);
xlabel('User Index');
ylabel('Mean SINR [dB]');
title('Average SINR per User');
grid on;
xline(NumGeoUser + 0.5, '--k', 'LineWidth', 1.2); 
text(NumGeoUser/2, max(meanSINR)+0.5, 'LEO Users', 'HorizontalAlignment', 'center');
text(NumGeoUser + NumLeoUser/2, max(meanSINR)+0.5, 'GEO Users', 'HorizontalAlignment', 'center');
ax=gca;
grid on;ax.Box = 'on';set(gca,'LooseInset',get(gca,'TightInset'),'FontSize',12);
ax.LineWidth = 1;  % Set box (axis) thickness to 1.5 points
saveLocation ="./";
Filename = fullfile(saveLocation, 'Figure2');
print(h_Fig, '-dpng','-r600',Filename)
