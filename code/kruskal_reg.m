function [beta0_final,beta_final,glmstats_final,dev_final] = ...
    kruskal_reg(X,M,y,r,dist,varargin)
% KRUSKAL_REG Fit rank-r Kruskal tensor regression
%
% INPUT:
%   X - n-by-p0 regular covariate matrix
%   M - array variates (or tensors) with dim(M) = [p1,p2,...,pd,n]
%   y - n-by-1 respsonse vector
%   r - rank of Kruskal tensor regression
%   dist - 'binomial', 'gamma', 'inverse gaussian','normal', or 'poisson'
%
% Output:
%   beta0_final - regression coefficients for the regular covariates
%   beta_final - a tensor of regression coefficientsn for array variates
%   glmstats_final - GLM regression summary statistics from last iteration
%   dev_final - deviance of final model

% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou (hua_zhou@ncsu.edu) and Lexin Li

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('M', @(x) isa(x,'tensor') || isnumeric(x));
argin.addRequired('y', @isnumeric);
argin.addRequired('r', @isnumeric);
argin.addRequired('dist', @(x) ischar(x));
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off')||strcmp(x,'iter'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-4, @(x) isnumeric(x) && x>0);
argin.addParamValue('Replicates', 5, @(x) isnumeric(x) && x>0);
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.parse(X,M,y,r,dist,varargin{:});

Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
TolFun = argin.Results.TolFun;
Replicates = argin.Results.Replicates;
wts = argin.Results.weights;

% check validity of rank r
if (isempty(r))
    r = 1;
elseif (r==0)
    [beta0_final,dev_final,glmstats_final] = ...
        glmfit_priv(X,y,dist,'constant','off');
    beta_final = 0;
    return;
end

% check dimensions
[n,p0] = size(X);
d = ndims(M)-1;             % dimension of array variates
p = size(M);                % sizes array variates
if (p(end)~=n)
    error('dimension of M does not match that of X!');
end
if (n<p0 || n<r*max(p(1:end-1)))    
    error('sample size n is not large enough to estimate all parameters!');
end

% turn off warnings
warning('off','stats:glmfit:IterationLimit');
warning('off','stats:glmfit:BadScaling');
warning('off','stats:glmfit:IllConditioned');

% pre-allocate variables
glmstats = cell(1,d+1);
dev_final = inf;

% convert M into a tensor T (if it's not)
TM = tensor(M);

% if space allowing, pre-compute mode-d matricization of TM
if (strcmpi(computer,'PCWIN64') || strcmpi(computer,'PCWIN32'))
    iswindows = true;
    % memory function is only available on windows !!!
    [dummy,sys] = memory; %#ok<ASGLU>
else
    iswindows = false;
end
% CAUTION: may cause out of memory on Linux
if (~iswindows || d*(8*prod(size(TM)))<.75*sys.PhysicalMemory.Available) %#ok<PSIZE>
    Md = cell(d,1);
    for dd=1:d
        Md{dd} = double(tenmat(TM,[d+1,dd],[1:dd-1 dd+1:d]));
    end
end

for rep=1:Replicates
    
    % initialize tensor regression coefficients from uniform [-1,1]
    beta = ktensor(arrayfun(@(j) 1-2*rand(p(j),r), 1:d, ...
        'UniformOutput',false));
    
    % main loop
    for iter=1:MaxIter
        % update coefficients for the regular covariates
        if (iter==1)
            [beta0, dev0] = glmfit_priv(X,y,dist,'constant','off', ...
                'weights',wts);
        else
            eta = Xj*beta{d}(:);
            [betatmp,devtmp,glmstats{d+1}] = ...
                glmfit_priv([X,eta],y,dist,'constant','off','weights',wts);
            beta0 = betatmp(1:end-1);
            % update scale of array coefficients and standardize
            beta = arrange(beta*betatmp(end));
            for j=1:d
                beta.U{j} = bsxfun(@times,beta.U{j},(beta.lambda').^(1/d));
            end
            beta.lambda = ones(r,1);
            % stopping rule
            diffdev = devtmp-dev0;
            dev0 = devtmp;
            if (abs(diffdev)<TolFun*(abs(dev0)+1))
                break;
            end
        end
        % cyclic update of the array coefficients
        eta0 = X*beta0;
        for j=1:d
            if (j==1)
                cumkr = ones(1,r);
            end
            if (exist('Md','var'))
                if (j==d)
                    Xj = reshape(Md{j}*cumkr,n,p(j)*r);
                else
                    Xj = reshape(Md{j}*khatrirao([beta.U(d:-1:j+1),cumkr]),...
                        n,p(j)*r);
                end
            else
                if (j==d)
                    Xj = reshape(double(tenmat(TM,[d+1,j],idxj))*cumkr, ...
                        n,p(j)*r);
                else
                    Xj = reshape(double(tenmat(TM,[d+1,j],idxj)) ...
                        *khatrirao({beta.U{d:-1:j+1},cumkr}),n,p(j)*r);
                end
            end
            [betatmp,dummy,glmstats{j}] = ...
                glmfit_priv([Xj,eta0],y,dist,'constant','off', ...
                'weights',wts); %#ok<ASGLU>
            beta{j} = reshape(betatmp(1:end-1),p(j),r);
            eta0 = eta0*betatmp(end);
            cumkr = khatrirao(beta{j},cumkr);
        end
    end
    
    % record if it has a smaller deviance
    if (dev0<dev_final)
        beta0_final = beta0;
        beta_final = beta;
        glmstats_final = glmstats;
        dev_final = dev0;
    end
    
    if (strcmpi(Display,'iter'))
        disp(' ');
        disp(['replicate: ' num2str(rep)]);
        disp([' iterates: ' num2str(iter)]);
        disp([' deviance: ' num2str(dev0)]);
        disp(['    beta0: ' num2str(beta0')]);
    end
    
end
warning on all;

end