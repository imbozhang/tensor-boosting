%% set tensor coefficient array

% b = imread('MPEG7_CE-Shape-1_Part_B/butterfly-1.gif');
% b = double(1-imresize(b,[64 64]));

load woman;                 % 256-by-256 woman image
b = X;
b = imresize(b, [16,16]);

b = b/max(abs(b(:)));
imagesc(b);
colormap(gray);

%% generate training and testing data sets

p0 = 5;
b0 = zeros(p0,1);
p1 = size(b,1);
p2 = size(b,2);

% generate covariates and response
n = 10000;
X = randn(n,p0);            % n-by-p regular design matrix
M = randn(p1,p2,n);         % p1-by-p2-by-n matrix variates
eta = X*b0 + squeeze(sum(sum(repmat(b,[1 1 n]).*M,1),2));
eta = (eta - mean(eta))/std(eta)*10;
prob = 1./(1+exp(-eta));
y = binornd(1,prob);

figure; scatter(eta,y);

% random split into training and test data
trainidx = rand(n,1)<0.1;
Xtrain = X(trainidx,:);
Mtrain = M(:,:,trainidx);
Ytrain = y(trainidx);
Xtest = X(~trainidx,:);
Mtest = M(:,:,~trainidx);
Ytest = y(~trainidx);

%% call boosting algorithm

r = 1;                      % rank for tensor regression
[trainerror,testerror,beta0,beta,glmstats] = ...
    tensor_logitboost([ones(size(Xtrain,1),1),Xtrain], ...
    Mtrain, Ytrain, r, 'Display','iter','BoostSteps',50, ...
    'Xtest', [ones(size(Xtest,1),1),Xtest], 'Mtest', Mtest, 'Ytest', Ytest, ...
    'Shrinkage', 0.1);

figure;
plot(1:length(trainerror),[trainerror; testerror]);
legend('training error', 'test error');
ylabel('misclassification error');