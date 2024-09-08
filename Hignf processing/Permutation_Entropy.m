 function [apen] = Permutation_Entropy(data,m,t)
 %∂‡≥ﬂ∂»Ïÿ
%  Calculate the permutation entropy
%  Input:   data: time series;
%           m: order of permuation entropy;
%           t: time delay of permuation entropy;
%  Output:  
%           apen: permuation entropy.
%Ref: G Ouyang, J Li, X Liu, X Li, Dynamic Characteristics of Absence EEG Recordings with Multiscale Permutation %  
%                             Entropy Analysis, Epilepsy Research, doi: 10.1016/j.eplepsyres.2012.11.003
%     X Li, G Ouyang, D Richards, Predictability analysis of absence seizures with permutation entropy, Epilepsy %  
%                            Research,  Vol. 77pp. 70-74, 2007
%  code is arranged by yyt in 2015.07   yangyuantaohit@163.com
N = length(data);
permlist = perms(1:m);
c(1:length(permlist))=0;
    
 for i=1:N-t*(m-1)
     [~,iv]=sort(data(i:t:i+t*(m-1)));
     for jj=1:length(permlist)
         if (abs(permlist(jj,:)-iv))==0
             c(jj) = c(jj) + 1 ;
         end
     end
 end

hist = c; 
c=hist(find(hist~=0));
p = c/sum(c);
pe = -sum(p .* log(p));
% normalized
apen=pe/log(factorial(m));
end