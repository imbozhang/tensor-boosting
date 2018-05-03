function [trainerror,testerror,beta0,beta,glmstats] = ...
    tensor_logitboost(X,M,y,r,varargin)
% TENSOR_LOGITBOOST Rank-r tesnor LogitBoosting
%
% INPUT:
%   X - n-by-p0 regular covariate matrix
%   M - array variates (or tensors) with dim(M) = [p1,p2,...,pd,n]
%   y - n-by-1 respsonse vector
%   r - rank of tensor regression
%
% Output:
%   trainerror - training error at each boosting steps
%   beta0 - regression coefficients for the regular covariates
%   beta - a tensor of regression coefficientsn for array variates
%   glmstats - GLM regression summary statistics from the last tensor
%       regression iterate in each boosting step

% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou, Bo Zhang and Lexin Li

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('M', @(x) isa(x,'tensor') || isnumeric(x));
argin.addRequired('y', @(x) all((x==0)|(x==1)));
argin.addRequired('r', @(x) isnumeric(x) && x>0);
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off')||strcmp(x,'iter'));
argin.addParamValue('BoostSteps', 10, @isnumeric);
argin.addParamValue('MaxIter', 20, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-4, @(x) isnumeric(x) && x>0);
argin.addParamValue('Replicates', 5, @(x) isnumeric(x) && x>0);
argin.addParamValue('Shrinkage', 1, @(x) isnumeric(x) && x>0);
argin.addParamValue('Xtest', [], @isnumeric);
argin.addParamValue('Mtest', [], @isnumeric);
argin.addParamValue('Ytest', [], @(x) isnumeric(x) || islogical(x));
argin.parse(X,M,y,r,varargin{:});

BoostSteps = argin.Results.BoostSteps;
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
TolFun = argin.Results.TolFun;
Replicates = argin.Results.Replicates;
Shrinkage = argin.Results.Shrinkage;
Xtest = argin.Results.Xtest;
Mtest = argin.Results.Mtest;
Ytest = argin.Results.Ytest;

% check training data
n = size(X,1);
p = size(M);    % sizes array variates
if (p(end)~=n)
    error('dimension of M does not match that of X!');
end

% Check testing data
if (isempty(Xtest) || isempty(Mtest) || isempty(Ytest))
    testdata = false;
    testerror = [];
else
    testdata = true;
    ntest = size(Xtest,1);
    if (size(Xtest,2)~=size(X,2))
        error('size of Xtest does not match that of X');
    end
    ptest = size(Mtest);
    if (~all(ptest(1:end-1)==p(1:end-1)))
        error('size of Mtest does not mathc that of M');
    end
    Mtest = tensor(Mtest);
end

% prepare storage arrays
beta0 = cell(1,BoostSteps);
beta = cell(1,BoostSteps);
glmstats = cell(1,BoostSteps);
trainerror = zeros(1,BoostSteps);
if (testdata)
    testerror = zeros(1,BoostSteps);
    Ftest = zeros(ntest,1);
end

% initialize boosting variables
F = zeros(n,1);
f = 4*(y-0.5);              % response for weighted tensor linear regression
wt = repmat(.25,n,1);       % weight for weighted tensor linear regression
if (strcmpi(Display,'iter'))
    disp(' ');
    disp('Boosting step: 0');
    disp(['Training error: ', num2str(nnz(y==0)/n,2)]);
    if (testdata)
        disp(['Testing error: ', num2str(nnz(Ytest==0)/ntest,2)]);
    end
end

% Boosting loop
for bs = 1:BoostSteps
    [beta0{bs},beta{bs},glmstats{bs}] = ...
        kruskal_reg(X,M,f,r,'normal','MaxIter',MaxIter,'TolFun',TolFun, ...
        'Replicates',Replicates,'weights',wt);
    erroridx = (y==1&F<0)|(y==0&F>=0);
    F = F + Shrinkage*(f-glmstats{bs}{end}.resid); % update regression function
    trainerror(bs) = nnz(erroridx)/n;
    if (testdata)
        etatest = Xtest*beta0{bs} + ...
            arrayfun(@(nn) innerprod(Mtest(:,:,nn),beta{bs}),(1:ntest)');
        Ftest = Ftest + Shrinkage*etatest;
        testerror(bs) = (nnz(Ytest==1&Ftest<0)+nnz(Ytest==0&Ftest>=0))/ntest; %#ok<AGROW>
    end
    if (strcmpi(Display,'iter'))
        disp(' ');
        disp(['Boosting step: ', num2str(bs)]);
        disp(['Training error: ', num2str(trainerror(bs),2)]);
        if (testdata)
            disp(['Testing error: ', num2str(testerror(bs),2)]);
        end
    end
    if (bs==BoostSteps), break, end;
    prob = 1./(1+exp(-F));          % update Bernoulli mean parameter
    wt = prob.*(1-prob);
    f = (y-prob)./wt;
end

end