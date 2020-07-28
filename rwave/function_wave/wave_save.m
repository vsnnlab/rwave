function [] = wave_save(env_dir,ii,state_ON,state_OFF)
ON_fire = state_ON == 2;
OFF_fire = state_OFF == 2;

fire_ind = find(sum(ON_fire,1) | sum(OFF_fire,1));
ind1 = min(fire_ind);
ind2 = max(fire_ind);

state_ON = state_ON(:,ind1:ind2);
state_OFF = state_OFF(:,ind1:ind2);

matfile = fullfile(env_dir,"wave"+num2str(ii)+".mat");
save(matfile,'state_ON','state_OFF');
disp("Wave states from " + num2str(ind1) + " to " + num2str(ind2) + " saved at " + matfile);
end