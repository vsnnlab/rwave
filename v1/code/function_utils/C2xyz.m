% Copyright (c) 2013, Chad Greene
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

function [x,y,z] = C2xyz(C)
% C2XYZ returns the x and y coordinates of contours in a contour
% matrix and their corresponding z values. C is the contour matrix given by 
% the contour function. 
% 
%
%% Syntax
% 
%  [x,y] = C2xyz(C)
%  [x,y,z] = C2xyz(C)
% 
%% Description 
% 
% [x,y] = C2xyz(C) returns x and y coordinates of contours in a contour
% matrix C
% 
% [x,y,z] = C2xyz(C) also returns corresponding z values. 
% 
% 
%% Example
% Given a contour plot, you want to know the (x,y) coordinates of the contours, 
% as well as the z value corresponding to each contour line. 
%
% C = contour(peaks); 
% [x,y,z] = C2xyz(C);
% 
% This returns 1 x numberOfContourLines cells of x values and y values, and
% their corresponding z values are given in a 1 x numberOfContourLines
% array. If you'd like to plot a heavy black line along all of the z=0
% contours and a dotted red line along the z = -2 contours, try this: 
% 
% hold on; % Allows plotting atop the preexisting peaks plot. 
% for n = find(z==0); % only loop through the z = 0 values. 
%     plot(x{n},y{n},'k','linewidth',2)
% end
% 
% for n = find(z==-2) % now loop through the z = -2 values. 
%     plot(x{n},y{n},'r:','linewidth',2)
% end
% 
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
% Created by Chad Greene, August 2013. 
% 
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
% See also contour, contourf, clabel, contour3, and C2xy.


m(1)=1; 
n=1;  
try
    while n<length(C)
        n=n+1;
        m(n) = m(n-1)+C(2,m(n-1))+1; 
        
    end
end

for nn = 1:n-2
     x{nn} = C(1,m(nn)+1:m(nn+1)-1); 
     y{nn} = C(2,m(nn)+1:m(nn+1)-1); 
     if nargout==3
        z(nn) = C(1,m(nn));
     end
end

end