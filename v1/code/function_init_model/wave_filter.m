function [Result] = wave_filter(pos,state,sig_wave)
state = double(gpuArray(state) == 2);
pos = gpuArray(pos);
dist_x = pos(:,1)-pos(:,1)';
dist_y = pos(:,2)-pos(:,2)';
dist2 = dist_x.^2+dist_y.^2;
gaussian = exp(-dist2/2/sig_wave^2);
filtered = gaussian*state;
Result = gather(filtered./max(filtered,1));
% figure;
% subplot(121); scatter(pos(:,1),pos(:,2),10,state(:,30),'filled');
% axis xy image; colormap(jet); colorbar;
% subplot(122); scatter(pos(:,1),pos(:,2),10,Result(:,30),'filled');
% axis xy image; colormap(jet); colorbar;
end