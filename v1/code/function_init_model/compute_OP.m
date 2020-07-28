function [Result] = compute_op(pos_ON,pos_OFF,w0_V1_ON,w0_V1_OFF,w_V1_ON,w_V1_OFF)
pos_ON = (pos_ON(:,1)+1i*pos_ON(:,2));
pos_OFF = (pos_OFF(:,1)+1i*pos_OFF(:,2));

% Initial
ON_loc = w0_V1_ON./sum(w0_V1_ON,2)*pos_ON;
OFF_loc = w0_V1_OFF./sum(w0_V1_OFF,2)*pos_OFF;
op0 = angle(ON_loc-OFF_loc);
% osi0 = abs(ON_loc-OFF_loc);
% ds0 = op0;

% Final
ON_loc = w_V1_ON./sum(w_V1_ON,2)*pos_ON;
OFF_loc = w_V1_OFF./sum(w_V1_OFF,2)*pos_OFF;
op = angle(ON_loc-OFF_loc);
% osi = abs(ON_loc-OFF_loc);
ds = op;

op0(op0<-pi/2) = op0(op0<-pi/2)+pi;
op0(op0>pi/2) = op0(op0>pi/2)-pi;
op(op<-pi/2) = op(op<-pi/2)+pi;
op(op>pi/2) = op(op>pi/2)-pi;
Result = [op0 op ds];
end