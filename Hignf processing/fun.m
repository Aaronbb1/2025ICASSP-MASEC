function fitness = fun(x,f,tau, DC, init, tol)
    x= round(x);
    alpha = x(1);        % moderate bandwidth constraint
    K = x(2);
%% 适应度函数
[u, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
%提取信号评价指标
% [m,n]=size(u);
% mm=2;
% Y=f;
% for ii=1:m
%   feature(ii)=SampEn(u(ii,:), mm, 0.2*std(u(ii,:)));
%      %Fuzzy_Entropy(mm,0.2*std(u(ii,:)),u(ii,:),1);%Fuzzy_Entropy( dim, r, data, tau )% %Approximate_Entropy(mm,0.2*std(u(ii,:)),u(ii,:),1);%SampEn(u(ii,:), mm, 0.2*std(u(ii,:)));%u的样本熵，0.2*std(imf1(1,:))表示求解样本熵r阀值，Sample_Entropy( mm,0.2*std(u(ii,:)),u(ii,:),1)
% end
% E = feature(1)+(sum(feature)-feature(1))/(m-1);
% X=sum(u);%☆☆    
% fit = (Y-X)/n;
% %pear=myPearson(X,Y);%归一化的皮尔逊函数☆☆
% % fit = feature(1)+(sum(feature)-feature(1))/(m-1);%☆☆  不需要管
% fitness = sum(fit)*E*10e5; 
% %% 样本熵
%     [u, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
%    % 提取信号评价指标
%     [m,n]=size(u);
%     mm=2;
%     Y=f;
%     for ii=1:m
%         feature(ii)=SampEn(u(ii,:), mm, 0.2*std(u(ii,:)));
%     %Fuzzy_Entropy(mm,0.2*std(u(ii,:)),u(ii,:),1);%Fuzzy_Entropy( dim, r, data, tau )% %Approximate_Entropy(mm,0.2*std(u(ii,:)),u(ii,:),1);%SampEn(u(ii,:), mm, 0.2*std(u(ii,:)));%u的样本熵，0.2*std(imf1(1,:))表示求解样本熵r阀值，Sample_Entropy( mm,0.2*std(u(ii,:)),u(ii,:),1)
%     end
%     % fitness= min((feature/pear)*D);%☆☆  不需要管
%  fitness= min(feature);
%% 包络熵
[u, ~,~] = VMD(f, alpha, tau, K, DC, init, tol);  
p=zeros(size(u));
E = zeros(size(u,1),1);
for i = 1:K
    a = abs(hilbert(u(i,:)));   % hilbert解调，abs() 函数返回数字的绝对值。
    p(i,:) = a./sum(a);  % 归一化
    E(i) = - sum(p(i,:).*log10(p(i,:)));%包络熵
end
fitness=min(E);
end


