function imf = decomposition_compilations(x,D_num)
% �ú����� ģ̬�ֽⷽ�� �ļ���
% ���в�������������õģ����������������רҵ���ݡ�֪ʶ���������á�
switch D_num
    case 1
        % EMD
        [imf,ort,nbits] = emd(x);
    case 2
        % TVF-EMD
        THRESH_BWR = 0.1;
        BSP_ORDER = 20;
        imf = tvf_emd(x,THRESH_BWR, BSP_ORDER);
    case 3
        % EEMD
        Nstd = 0.4;
        NE = 200;
        numImf = 9; % �ֽ�imf����
        imf=eemd(x,Nstd,NE);

    case 4
        % VMD
        K = 6; % �ֽ�imf����
        tau = 1;
        alpha = 2000;
        DC=0; init=1;tol= 1e-7;
        [imf, imf_hat, omega] = VMD(x, alpha, tau, K, DC, init, tol);
    case 5
        %  CEEMDAN
        Nstd = 0.4;
        NR = 10;
        MaxIter = 50;
        [imf, its]=ceemdan(x,Nstd,NR,MaxIter);
    case 6
        % LMD
        imf = LMD(x);
    case 7
        % ITD
        N_max = 10;
        imf=ITD(x,N_max);
    case 8
        % SVMD: https://www.sciencedirect.com/science/article/pii/S0165168420301535?via%3Dihub
        maxAlpha=20000;
        tau=0;
        stopc=4;
        tol=1e-6;
        [imf,imf_hat,omega]=svmd(x,maxAlpha,tau,tol,stopc);
    case 9
        %  ICEEMDAN: https://www.sciencedirect.com/science/article/pii/S1746809414000962#sec0015
        Nstd = 0.4;
        NR = 10;
        MaxIter = 50;
        [imf,its]=iceemdan(x,Nstd,NR,MaxIter);
    case 10
        % FMD: https://ieeexplore.ieee.org/document/9732251
        filtersize = 30;
        cutnum = 7;
        modenum = 5;
        maxiternum = 20;
        fs = 2e4;
        imf = FMD(fs, x, filtersize, cutnum, modenum, maxiternum);

     case 11
         % REMD
         imf = emd_sssc(x);
    case 12
        % SGMD
        fs = 2e4;
        imf = SGMD(x,fs,1,0.95,0.01);
    case 13
        % RLMD
        [imf, ams, fms, ort] = robust_lmd(x);

    case 14
        % ESMD
        delt_t = 0.5;
        minLoop=1;
        maxLoop=40;
        extremeNumR=4; % >=4
        imf =ESMD(x,delt_t,minLoop, maxLoop, extremeNumR);
    case 15
        % ceemd
        Nstd = 0.2;
        NE = 8;
        TNM = 4;
        imf=ceemd(x,Nstd,NE,TNM);
    case 16
        % SSA
        M=5;
        imf=SSA(x,M);
    case 17
        % EWT
        imf = ewt1d(x);
    case 18
        % SWD
        cmps_thresh  = 0.025;
        detail_depth = 1e-5;
        imf = SwD_v2(x, cmps_thresh, detail_depth);
    case 19
        % MODWT
        wname='db4';
        imf = modwt(x,wname);
    case 20
        % RPSEMD
        imf = Rpsemd(x);  

end
% �� imf��ʽͳһ��ʽ��ģ̬���� x ���ݳ���
% ���� imf(1, : )��IMF1��imf(end , : )��Ϊ�в�

if size(imf,1)>size(imf,2)
    imf = imf';
end
end