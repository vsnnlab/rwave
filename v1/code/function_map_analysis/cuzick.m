function [p] = cuzick(x,varargin)
% CUZICK: Perform the Cuzick's test on trend.
% This function provides a Wilcoxon-type test for trend across a group of
% three or more independent random samples.
% Assumptions:
% - Data must be at least ordinal
% - Groups must be selected in a meaningful order i.e. ordered
% If you do not choose to enter your own group scores then scores are
% allocated uniformly (1 ... n) in order of selection of the n groups.
% The null hypothesis of no trend across the groups T will have mean E(T),
% variance var(T) and the null hypothesis is tested using the normalised
% test statistic z.
% A logistic distribution is assumed for errors. Please note that this test
% is more powerful than the application of the Wilcoxon rank-sum /
% Mann-Whitney test between more than two groups of data.
% Cuzick J. A Wilcoxon-Type Test for Trend. Statistics in Medicine
% 1985;4:87-89.
%
% Syntax: 	cuzick(x,score)
%      
%     Inputs:
%           X - Nx2 data matrix 
%           SCORE - order of selection of the groups - default 1:1:max(x(:,2))
%     Outputs:
%           - Cuzick's statistics and p-value
%
%   Example:
% Mice were inoculated with cell lines, CMT 64 to 181, which had been
% selected for their increasing metastatic potential. The number of lung
% metastases found in each mouse after inoculation are quoted below:
%
%                                 Sample
%                   ---------------------------------
%                      64   167  170  175  181
%                   ---------------------------------
%                      0    0    2    0    2
%                      0    0    3    3    4
%                      1    5    6    5    6
%                      1    7    9    6    6
%                      2    8    10   10   6
%                      2    11   11   19   7
%                      4    13   11   56   18
%                      9    23   12   100  39    
%                           25   21   132  60
%                           97
%                   ---------------------------------
%
%       Data matrix must be:
%    d=[0 0 1 1 2 2 4 9 0 0 5 7 8 11 13 23 25 97 2 3 6 9 10 11 11 12 21 ...
%       0 3 5 6 10 19 56 100 132 2 4 6 6 6 7 18 39 60];
%    g=[ones(1,8) 2.*ones(1,10) 3.*ones(1,9) 4.*ones(1,9) 5.*ones(1,9)];
%    x=[d' g'];
%
%           Calling on Matlab the function: cuzick(x)
% (in this case, the groups are automated scored from 1 to 5)
%
%           Answer is:
%
% CUZICK'S TEST FOR NON PARAMETRIC TREND ANALYSIS
% --------------------------------------------------------------------------------
%     Group    Score    Samples    Ranks_sum
%     _____    _____    _______    _________
% 
%     1        1         8            79    
%     2        2        10           256    
%     3        3         9           229    
%     4        4         9         246.5    
%     5        5         9         224.5    
% 
% Ties factor: 366
% --------------------------------------------------------------------------------
%  
% CUZICK'S STATISTICS
% --------------------------------------------------------------------------------
%      L       T        E       Var       z       one_tailed_p_values
%     ___    ______    ____    _____    ______    ___________________
% 
%     136    3386.5    3128    14973    2.1125    0.01732   
%                         
% With these data we are interested in a trend in one direction only,
% therefore, we can use a one sided test for trend. We have shown a
% statistically significant trend for increasing number of metastases across
% these malignant cell lines in this order.
%
%           Created by Giuseppe Cardillo
%           giuseppe.cardillo-edta@poste.it
%
% To cite this file, this would be an appropriate format:
% Cardillo G. (2008) Cuzick's test: A Wilcoxon-Type Test for Trend
% http://www.mathworks.com/matlabcentral/fileexchange/22059

%Input Error handling
p = inputParser;
addRequired(p,'x',@(x) validateattributes(x,{'numeric'},{'real','finite','nonnan','nonempty','ncols',2}));
addOptional(p,'score',[],@(x) isempty(x) || (all(isnumeric(x(:))) && isrow(x) && all(isreal(x(:))) && all(isfinite(x(:))) && ~all(isnan(x(:))) && all(x(:)>0) && all(fix(x(:))==x(:))));
parse(p,x,varargin{:});
assert(all(x(:,2) == fix(x(:,2))),'Warning: all elements of column 2 of input matrix must be whole numbers')
score=p.Results.score;
clear p
k=max(x(:,2)); %number of groups
if isempty(score) %check score
   score=1:1:k;
end

tr=repmat('-',1,80);% set divisor
disp('CUZICK''S TEST FOR NON PARAMETRIC TREND ANALYSIS')
disp(tr)
ni=crosstab(x(:,2)); %elements for each group
N=sum(ni); %total elements
R=ones(1,k); L=R; T=R; %vectors preallocation
[r,t]=tiedrank(x(:,1)); %ranks and ties

for I=1:k
    R(I)=sum(r(x(:,2)==I)); %sum of ranks of each group
    L(I)=score(I)*ni(I);
    T(I)=score(I)*R(I);
end
disp(table((1:k)',score',ni,R','VariableNames',{'Group','Score','Samples','Ranks_sum'}))
if logical(t) %if there are ties
    fprintf('Ties factor: %d\n',2*t)
end
disp(tr); disp(' ')
clear idx r %clear unnecessary variables

%For the null hypothesis of no trend across the groups;
%T will have mean E(T), variance var(T) and the null hypothesis is tested
%using the normalised test statistic z.
Lval=sum(L); Tval=sum(T);
Et=Lval*(N+1)/2; %mean of T
Vart=((N*sum(score.*L)-Lval^2)*(N+1)/12)-t/6; %Variance of T
z=abs(Tval-Et)/sqrt(Vart); %z statistic
p=1-0.5*erfc(-double(z)/realsqrt(2)); %p-value
disp('CUZICK''S STATISTICS')
disp(tr)
disp(table(Lval,Tval,Et,Vart,z,p,'VariableNames',{'L','T','E','Var','z','one_tailed_p_values'}))