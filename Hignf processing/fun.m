function fitness = fun(x,f,tau, DC, init, tol)
    x= round(x);
    alpha = x(1);        % moderate bandwidth constraint
    K = x(2);
%% ��Ӧ�Ⱥ���
[u, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
%��ȡ�ź�����ָ��
% [m,n]=size(u);
% mm=2;
% Y=f;
% for ii=1:m
%   feature(ii)=SampEn(u(ii,:), mm, 0.2*std(u(ii,:)));
%      %Fuzzy_Entropy(mm,0.2*std(u(ii,:)),u(ii,:),1);%Fuzzy_Entropy( dim, r, data, tau )% %Approximate_Entropy(mm,0.2*std(u(ii,:)),u(ii,:),1);%SampEn(u(ii,:), mm, 0.2*std(u(ii,:)));%u�������أ�0.2*std(imf1(1,:))��ʾ���������r��ֵ��Sample_Entropy( mm,0.2*std(u(ii,:)),u(ii,:),1)
% end
% E = feature(1)+(sum(feature)-feature(1))/(m-1);
% X=sum(u);%���    
% fit = (Y-X)/n;
% %pear=myPearson(X,Y);%��һ����Ƥ��ѷ�������
% % fit = feature(1)+(sum(feature)-feature(1))/(m-1);%���  ����Ҫ��
% fitness = sum(fit)*E*10e5; 
% %% ������
%     [u, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
%    % ��ȡ�ź�����ָ��
%     [m,n]=size(u);
%     mm=2;
%     Y=f;
%     for ii=1:m
%         feature(ii)=SampEn(u(ii,:), mm, 0.2*std(u(ii,:)));
%     %Fuzzy_Entropy(mm,0.2*std(u(ii,:)),u(ii,:),1);%Fuzzy_Entropy( dim, r, data, tau )% %Approximate_Entropy(mm,0.2*std(u(ii,:)),u(ii,:),1);%SampEn(u(ii,:), mm, 0.2*std(u(ii,:)));%u�������أ�0.2*std(imf1(1,:))��ʾ���������r��ֵ��Sample_Entropy( mm,0.2*std(u(ii,:)),u(ii,:),1)
%     end
%     % fitness= min((feature/pear)*D);%���  ����Ҫ��
%  fitness= min(feature);
%% ������
[u, ~,~] = VMD(f, alpha, tau, K, DC, init, tol);  
p=zeros(size(u));
E = zeros(size(u,1),1);
for i = 1:K
    a = abs(hilbert(u(i,:)));   % hilbert�����abs() �����������ֵľ���ֵ��
    p(i,:) = a./sum(a);  % ��һ��
    E(i) = - sum(p(i,:).*log10(p(i,:)));%������
end
fitness=min(E);
end


