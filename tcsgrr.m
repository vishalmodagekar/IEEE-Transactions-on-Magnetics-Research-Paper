% Tilted Correlation Screening with Generalized Ridge Regression (TCS-GRR) Algorithm
% Author: Vishal Modagekar (vcmodagekar@gmail.com)

function [ active, thr_seq, tc_seq, beta, bic_seq, best_subset ] = tcsgrr( X, y, op, rpm, thr_rep, max_size, max_iter, thr_step, eps )
% Tilted Correlation Screening with Generalized Ridge Regression (TCS-GRR) Algorithm
% Ref: V. Modagekar et. al., "Optimization in Tri-Axial Degaussing System Design and Estimation of Degaussing Coil Currents," 
% in IEEE Transactions on Magnetics, vol. 53, no. 4, pp. 1-12, April 2017.
% Input:
% X -- Design matrix.
% y -- Response vector.
% op -- For op==1 tilted correlation coincides with OLS regression coefficient when regressing residual response onto covariate. 
%       For op==2, tilted correlation coincides with sample partial correlation between covariate and residual response given other 
%       covariates up to a constant multiplicative factor, i.e., L2 norm of residual response.
%       Default is set to 1.
% rpm -- Multiplier of Ridge parameters, default is set to 1.
% thr_rep -- The number of times for which the threshold selection procedure is repeated, default is set to 1.
% max_size -- The maximum number of the variables conditional on which the contribution of each variable 
%             to the response is measured, when not given default is set to be half the number of observations.
% max_iter -- Maximum number of iterations, default is set to number of columns in the design matrix.
% eps -- Effective zero, default is set to 1e-10.
% thr_step -- A step size used for threshold selection. When not given, it is chosen automatically.
%
% Output:
% active -- A sequence of indices that indicate the order of variable inclusions.
% thr_seq -- A sequence of thresholds selected over the iterations.
% tc_seq -- A sequence of tilted correlations of variables selected over the iterations.
% beta -- History of estimated regression coefficients of variables selected over the iterations. Each 
%         row gives the regression coefficients of variables at particular iteration. 
% bic_seq -- Extended Bayesian Information Criterion (BIC) computed over the iterations.
% best_subset -- Finally chosen variables using the extended BIC.

% Copy of unprocessed design matrix and target
X_ut = X; y_ut = y;

% Get dimensions of design matrix
[n,p] = size(X);

% Set default values of inputs if not given
if nargin<3 , op=1; end
if nargin<4 , rpm=1; end
if nargin<5 , thr_rep=1; end
if nargin<6 , max_size = floor(n/2); end
if nargin<7 , max_iter = p; end
if nargin<8 , thr_step=[]; end
if nargin<9 , eps =1e-10; end 

if(~isempty(thr_step))
    step = thr_step;
end

% Normalize design matrix and target
colnorm = sqrt(ones(1,size(X,1))*X.^2);
X = normalize(X);
ynorm = norm(y); y = y/ynorm ;

% Initialize and pre-allocated required objects
active = []; inactive = setdiff(1:p, active);
Z = X; res = y; 
thr_seq = zeros(p-1,1);
tc_seq = zeros(p,1);

% Pre-allocate matrix for regression coefficients history
beta = zeros(p+1,p);

% Execute the algorithm for given iterations
for iter_num = 2:max_iter+1
    C = Z'*Z;
    if(length(inactive)==1)
        active = [active,inactive];
        inactive = setdiff(1:p, active);
        tc_seq(iter_num-1) = abs(Z'*res/norm(res));
    elseif(length(inactive)==2)
        thr = median(thr_seq); 
        thr_seq(iter_num-1) = thr;
        if(abs(C(~tril(C))) <= thr)
             corr = Z'*res/norm(res);
             
             % Find variable in inactive set having maximum marginal
             % correlation with residual
             k = find(abs(corr) == max(abs(corr)));
             active = [active,inactive(k)];
             inactive = setdiff(1:p, active);
             tc_seq(iter_num-1) = max(abs(corr));
        else
            % Find tilted correlation according to 'op'
            tc = zeros(2,1);
             a1 = (Z(:,2)*((Z(:,2)'*Z(:,2))^(-1)*Z(:,2)'*Z(:,1)))'*Z(:,1);
             a2 = (Z(:,1)*((Z(:,1)'*Z(:,1))^(-1)*Z(:,1)'*Z(:,2)))'*Z(:,2);
             if(op==1)
                 tc(1) = (Z(:,1) - Z(:,2)*((Z(:,2)'*Z(:,2))^(-1)*Z(:,2)'*Z(:,1)))'*res/(1-a1);
                 tc(2) = (Z(:,2) - Z(:,1)*((Z(:,1)'*Z(:,1))^(-1)*Z(:,1)'*Z(:,2)))'*res/(1-a2);
             else
                 ay1 = (Z(:,2)*((Z(:,2)'*Z(:,2))^(-1)*Z(:,2)'*res))'*res/(res'*res);
                 ay2 = (Z(:,1)*((Z(:,1)'*Z(:,1))^(-1)*Z(:,1)'*res))'*res/(res'*res);
                 tc(1) = (Z(:,1) - Z(:,2)*((Z(:,2)'*Z(:,2))^(-1)*Z(:,2)'*Z(:,1)))'*res/sqrt((1-a1)*(1-ay1));
                 tc(2) = (Z(:,2) - Z(:,1)*((Z(:,1)'*Z(:,1))^(-1)*Z(:,1)'*Z(:,2)))'*res/sqrt((1-a2)*(1-ay2));
             end
             k = find(abs(tc) == max(abs(tc)));
             active = [active,inactive(k)];
             inactive = setdiff(1:p, active);
             tc_seq(iter_num-1) = max(abs(tc));
         end
    else
         % Set step size for threshold selection if not given
         if(isempty(thr_step)) 
              step = max(1, floor((p-length(active))*(p-1-length(active))/(n-length(active))/10));
         end
         
         % Get threshold to identify highly correlated covariates
         thr = get_thr(C, n, p-length(active), thr_rep, step);
         thr_seq(iter_num-1) = thr;
         
         % Hard threshold the correlation matrix of inactive covariates
         C1 = thresh(C, thr, eps);
         corr = Z'*res/norm(res);
         k = find(abs(corr) == max(abs(corr)));
         
         % Find variables having correlation more than threshold with variable having maximum marginal correlation with residual
         hcv = setdiff(1:size(Z,2), k);
         hcv = hcv(abs(C1(setdiff(1:size(C1,1),k),k))>0);
         
         % Add variable having maximum tilted correlation with the residual in the active set
         if (isempty(hcv))
             active = [active,inactive(k)];
             inactive = setdiff(1:p, active);
             tc_seq(iter_num-1) = max(abs(corr));
         else
             J = [k, hcv]; tc = [];
             m = zeros(1,length(J));
             for it = 1:length(J)
                  m(it) = sum(C1(J(it),:)~=0)-1;
             end
             m = max(m);
             m = max(2, min(max_size, m));
             for i = J
                 z = Z(:,i);   
                 hcv = setdiff(1:size(Z,2), i);
                 hcv = hcv(abs(C1(setdiff(1:size(C1,1),i),i))>0);
                 num = length(hcv);

                 % Discard variables from 'hcv' such that number of 'hcv' don't exceed 'max_size' 
                 if (num>m)
                    if(i~=k)
                       hcv = hcv(hcv~=k);
                       [~, iseq] = sort(abs(C(hcv,i)), 'descend');
                       hcv = [k, hcv(iseq(1:m-1))];
                    else
                       [~, iseq] = sort(abs(C(hcv,i)), 'descend'); 
                       hcv = hcv(iseq(1:m));
                    end
                 else   
                   [~, iseq] = sort(abs(C(hcv,i)), 'descend');
                   hcv = hcv(iseq(1:length(iseq)));
                 end
                 hcv_len = length(hcv);

                 % Remove variables such that their correlation matrix is positive semidefinite
                 while (hcv_len>0 && min(eig(C([hcv,i],[hcv,i])))<eps)
                       hcv = hcv(1:hcv_len-1);
                       hcv_len = hcv_len-1;
                 end
                 if (hcv_len==0)
                    tc = [tc, 0];
                    continue;
                 end 

                 % Find tilted correlation according to 'op'
                 a = (Z(:,hcv)*((Z(:,hcv)'*Z(:,hcv))^(-1)*Z(:,hcv)'*z))'*z;
                 if (op==1)
                    tc = [tc, (z - Z(:,hcv)*((Z(:,hcv)'*Z(:,hcv))^(-1)*Z(:,hcv)'*z))'*res/(1-a)];
                 else
                    ay = (Z(:,hcv)*((Z(:,hcv)'*Z(:,hcv))^(-1)*Z(:,hcv)'*res))'*res/(res'*res);
                    tc = [tc, (z - Z(:,hcv)*((Z(:,hcv)'*Z(:,hcv))^(-1)*Z(:,hcv)'*z))'*res/sqrt((1-a)*(1-ay))];
                 end
             end

             % Terminate if all tilted correlations of variables in 'J' are 0
             if(sum(tc~=0)==0) , break; end

             % Find variable having maximum tilted correlation with residual and include it in the active set
             [tc_sort, itc_sort] = sort(abs(tc), 'descend');
             k = inactive(J(itc_sort(1)));
             active = [active, k];
             inactive = setdiff(1:p, active);
             tc_seq(iter_num-1) = tc_sort(1);
        end
    end
    
    % Terminate if condition number of active variables correlation matrix tends to infinity
    active_cor_eig = eig(X(:,active)'*X(:,active));
    if(abs(max(active_cor_eig)/min(active_cor_eig))>1/eps) 
        active = active(1:length(active)-1);
        break; 
    end
    
    % Find normalized version of regression coefficients as algorithm iterates
    beta_tmp = zeros(p,1);
    re = grr(X(:, active), res, rpm);              
    beta_tmp(active) = beta(iter_num-1, active)' + re;
    beta(iter_num,:) = beta_tmp;
    
    % Update residual and design matrix for next iteration
    res = y - X*beta_tmp;
    if (~isempty(inactive))
        Z = X(:,inactive) - X(:,active)*((X(:,active)'*X(:,active))^(-1)*X(:,active)'*X(:,inactive));
        Z = normalize(Z);
    end
end

% Denormalize and rescale the regression coefficients history 
beta = beta*diag(1./colnorm);
beta = ynorm*beta;
beta = beta(2:length(active)+1,:);

% Calculate actual residual from target for all iterations 
res_ut = y_ut*ones(1,length(active)) - X_ut*beta';

% Calculate extended BIC for sequence of models
bic_seq = log(sum(res_ut.^2)/n)+(1:length(active))/n*(log(n)+2*log(p)); 
best_subset = active(1:find(bic_seq == min(bic_seq)));

% Clip sequence of threshold correlation and tilted correlation of active variables according to the iterations executed
if(length(active)~=p)
    thr_seq = thr_seq(1:length(active));
    tc_seq = tc_seq(1:length(active));
end

%%%%%%%%%% Estimate the generalized ridge regression coefficients %%%%%%%%%
function beta = grr( X, y, rpm )
ynorm = norm(y); y = y/ynorm;
[n, p] = size(X);
[V, D2] = eig(X'*X);
Z = X*V;
d2 = diag(D2);
alpOLS = D2^(-1)*Z'*y;
sigsq = (y'*y - alpOLS'*Z'*y)/(n-p-1);

% Diagonal matrix containing ridge parameters
K = rpm*(2*sigsq/max(d2))*(diag(alpOLS)^-2);
alpGR = (eye(p) - K*(D2 + K)^-1)*alpOLS;
betGR = V*alpGR;
beta = ynorm*betGR;

%%%%%%%%%%%%%%% Normalize columns of X to have length one %%%%%%%%%%%%%%%%%
function nX = normalize(X)
nX = X*diag(1./sqrt(ones(1,size(X,1))*X.^2));

%%%%%%%%%%%% Select a threshold for sample correlation matrix %%%%%%%%%%%%%
function thr = get_thr( C, n, p, thr_rep, thr_step )
corr = abs(C(~tril(ones(size(C))))); 
thr_seq = zeros(1:thr_rep);
d = p*(p-1)/2;
nu = min(0.05, 1/sqrt(p));
sc = sort(corr);
for iter = 1:thr_rep
    D = mvnrnd(zeros(p,1),diag(ones(p,1)),n);
    D = normalize(D);
    ref = abs(D(~tril(ones(size(D)))));
    i = 1;

    % Find p-values to get threshold signifying high marginal correlation
    while (i<=d)
        c = sc(i);
        prob = sum(ref>c)/d;
        if(prob<=(d-i+1)/d*nu) , break; end
        i = i + thr_step;  
    end
    thr_seq(iter) = c;
end
thr = median(thr_seq);

%%%%%%%%%%%%%%%%%%%%%%%% Hard-threshold a matrix %%%%%%%%%%%%%%%%%%%%%%%%%%
function C1 = thresh( C, thr, eps )
C1=C;
C1(abs(C1)<thr+eps)=0;

