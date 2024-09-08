function msen=MsEn(m,r,Xn,t)

%����������ʱ�����еĶ�߶���
%input
%  m��ά��
%  r����ֵ
%  Xn��ʱ������
%  t���߶�����
%output
%  msen���������õ�������ֵ

M=fix(length(Xn)/t);
for i=1:M
       Yn(i)= sum ( Xn((((i-1)*t)+1) :(i*t) ) )/ t ; 
end
for j=1:(length(Yn)-m+1)
    Ym(:,j)=Yn(((j-1)+1):(j-1+m));
end
for k=1:(length(Yn)-m)
    for g=(k+1):(length(Yn)-m+1)
        mm=abs(Ym(:,k)-Ym(:,g));
        d(g,k)=max(mm);
    end
end
Bm=0;

for k=1:(length(Yn)-m)
    i=0;
    for g=(k+1):(length(Yn)-m+1)
        if d(g,k)<r
           i=i+1; 
           Bm(k)=i;
        end
    end
end
Cmr=Bm/(M-m);
meanCmr=mean(Cmr);
for j=1:(length(Yn)-m)
    Ym1(:,j)=Yn(((j-1)+1):(j+m));
end
for k=1:(length(Yn)-m-1)
    for g=(k+1):(length(Yn)-m)
        d1(g,k)=max(abs(Ym1(:,k)-Ym1(:,g)));
    end
end
Bm1=0;
for k=1:(length(Yn)-m-1)
    i=0;
    for g=(k+1):(length(Yn)-m)
        if d1(g,k)<r
            i=i+1;
            Bm1(k)=i;
        end
    end
end
Cmr1=Bm1/(M-m+1);
meanCmr1=mean(Cmr1);
msen=-log(meanCmr1/meanCmr);
