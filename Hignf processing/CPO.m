% %  Crested Porcupine Optimizer: A new nature-inspired metaheuristic % % %
function [Gb_Fit,Gb_Sol,Conv_curve]=CPO(Pop_size,Tmax,lb,ub,dim,fobj)

%%%%-------------------Definitions--------------------------%%
%%
Conv_curve=zeros(1,Tmax);
ub=ub.*ones(1,dim);
lb=lb.*ones(1,dim);
%%-------------------Controlling parameters--------------------------%%
%%
N=Pop_size; %% Is the initial population size.
N_min=round(0.8*Pop_size); %% Is the minimum population size.
T=2; %% The number of cycles
alpha=0.2; %% The convergence rate
Tf=0.8; %% The percentage of the tradeoff between the third and fourth defense mechanisms

%%---------------Initialization----------------------%%
%%
X=initialization(Pop_size,dim,ub,lb); % Initialize the positions of crested porcupines
t=0; %% Function evaluation counter 
%%---------------------Evaluation-----------------------%%
for i=1:Pop_size
    fitness(i)=fobj(X(i,:));
end
% Update the best-so-far solution
[Gb_Fit,index]=min(fitness);
Gb_Sol=X(index,:);    
%% A new array to store the personal best position for each crested porcupine
Xp=X;

%%  Optimization Process of CPO
while t<=Tmax
    t
    r2=rand;
    for i=1:Pop_size
        
        U1=rand(1,dim)>rand;
        if rand<rand %% Exploration phase
            if rand<rand %% First defense mechanism
                %% Calculate y_t
                y=(X(i,:)+X(randi(Pop_size),:))/2;
                X(i,:)=X(i,:)+(randn).*abs(2*rand*Gb_Sol-y);
            else %% Second defense mechanism
                y=(X(i,:)+X(randi(Pop_size),:))/2;
                X(i,:)=(U1).*X(i,:)+(1-U1).*(y+rand*(X(randi(Pop_size),:)-X(randi(Pop_size),:)));
            end
        else
             Yt=2*rand*(1-t/(Tmax))^(t/(Tmax));
             U2=rand(1,dim)<0.5*2-1;
             S=rand*U2;
            if rand<Tf %% Third defense mechanism
                %% 
                St=exp(fitness(i)/(sum(fitness)+eps)); % plus eps to avoid division by zero
                S=S.*Yt.*St;
                X(i,:)= (1-U1).*X(i,:)+U1.*(X(randi(Pop_size),:)+St*(X(randi(Pop_size),:)-X(randi(Pop_size),:))-S); 

            else %% Fourth defense mechanism
                Mt=exp(fitness(i)/(sum(fitness)+eps));
                vt=X(i,:);
                Vtp=X(randi(Pop_size),:);
                Ft=rand(1,dim).*(Mt*(-vt+Vtp));
                S=S.*Yt.*Ft;
                X(i,:)= (Gb_Sol+(alpha*(1-r2)+r2)*(U2.*Gb_Sol-X(i,:)))-S; 
            end
        end
        %% Return the search agents that exceed the search space's bounds
        for j=1:size(X,2)
            if  X(i,j)>ub(j)
                X(i,j)=lb(j)+rand*(ub(j)-lb(j));
            elseif  X(i,j)<lb(j)
                X(i,j)=lb(j)+rand*(ub(j)-lb(j));
            end

        end  
         % Calculate the fitness value of the newly generated solution
         nF=fobj(X(i,:));
        %% update Global & Personal best solution
        if  fitness(i)<nF
            X(i,:)=Xp(i,:);    % Update local best solution
        else
            Xp(i,:)=X(i,:);
            fitness(i)=nF;
            %% update Global best solution
            if  fitness(i)<=Gb_Fit
                Gb_Sol=X(i,:);    % Update global best solution
                Gb_Fit=fitness(i);
            end
        end

    end
    Pop_size=fix(N_min+(N-N_min)*(1-(rem(t,Tmax/T)/Tmax/T)));
    t=t+1; % Move to the next generation
    if t>Tmax
        break
    end
    Conv_curve(t)=Gb_Fit;
 end
end