%%% 模糊熵计算函数 %%% 
function SampEn = Fuzzy_Entropy( dim, r, data, tau )
% FUZZYEN Fuzzy Entropy
%   calculates the fuzzy entropy of a given time series data

% Similarity definition based on vectors' shapes, together with the
% exclusion of self-matches, earns FuzzyEn stronger relative consistency
% and less dependence on data length.

%   dim     : embedded dimension 
%   r       : tolerance (typically 0.2 * std)
%   data    : time-series data
%   tau     : delay time for downsampling (user can omit this, in which case
%             the default value is 1)
%

if nargin < 4, tau = 1; end
if tau > 1, data = downsample(data, tau); end

N = length(data);
result = zeros(1,2);

for m = dim:dim+1% 该循环用于实现算法的第六步
    Bi = zeros(1,N-m+1);
    dataMat = zeros(m,N-m+1);
    
    % setting up data matrix
    for i = 1:m
        dataMat(i,:) = data(i:N-m+i);
    end
    
    % counting similar patterns using distance calculation
    for j = 1:N-m+1
        % calculate Chebyshev distance, excluding self-matching case
        dist = max(abs(dataMat - repmat(dataMat(:,j),1,N-m+1)));
        % calculate Heaviside function of the distance
        % User can change it to any other function
        % for modified sample entropy (mSampEn) calculation
        D = (dist <= r);
        % excluding self-matching case
        Bi(j) = (sum(D)-1)/(N-m);
    end
    
    % summing over the counts
    result(m-dim+1) = sum(Bi)/(N-m+1);
    
end

SampEn = -log(result(2)/result(1));

end
