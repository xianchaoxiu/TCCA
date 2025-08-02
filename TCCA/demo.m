
clc;close all; clear all; warning off
load('Caltech101-7.mat');
%set(0,'defaultaxesfontsize',10);
% set(0,'defaulttextfontsize',10);
label=Y;
%size(label)

for i1=2:2:20
     %for i1=12:2:20
      Dim=i1;
%for i1=16
filename = strcat('redata\', 'calre', num2str(i1,'%01d'),'.txt');
    ordata1 = load(filename);
   
    
    A=ordata1(:,1:Dim);
    B=ordata1(:,(Dim+1):2*Dim);
    C=ordata1(:,(2*Dim+1):3*Dim);
    D=ordata1(:,(3*Dim+1):4*Dim);
    option.n=Dim;option.p=Dim;option.q=Dim;option.L=Dim;option.l=Dim;
    option.t1=0.01;option.t2=0.01;option.t3=0.01;option.t4=0.01;
    option.b1=0.5;option.b2=0.4;option.b3=0.45;option.b4=0.3;
    option.X0=eye(Dim,Dim);option.Y0=eye(Dim,Dim);option.Z0=eye(Dim,Dim);option.S0=eye(Dim,Dim);
    option.maxiter=10;option.epsilon=0.1;
   
    option.inner_tol=100;option.tol=0.0001; 
    nExperiment=10;

    
    results_TCCAO=zeros(nExperiment, 4);
    results_TCCAOS=zeros(nExperiment, 4);
   
    %K=[1,2,3];
    testRatio=0.3;
    train_num = 1032;
    test_num = 442;
    class1=7;
%   option.b1=0.8;option.b2=0.9;option.b3=0.9;

   for iExperiment = 1:nExperiment 
    fprintf('=========================================================\n');
     %%%%%%%%%%%%%%%%%%%%%%%%%%%
     %%%%%%%%%% TCCAO %%%%%%%%%
     %%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    [Z_TCCAO,~]=mvdr_ttcca({A,B,C,D},'d',i1,'maxiters',10000,'epsilon',0.0001);
     t_TCCAO=toc;
      [accr_TCCAO, f1score_TCCAO,~] = knnacc4(testRatio,train_num,test_num,Z_TCCAO,label,class1);
    dataformat_TCCAO = '%d-th experiment:  accr_TCCAO = %f, f1score_TCCAO = %f,time_TCCAO=%f\n';
    dataValue_TCCAO = [iExperiment, accr_TCCAO, f1score_TCCAO, t_TCCAO];
    results_TCCAO(iExperiment, :)=dataValue_TCCAO;
    fprintf(dataformat_TCCAO, dataValue_TCCAO);
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
     %%%%%%%%%% TCCAOS %%%%%%%%%
     %%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
mvdr_ttcca({A,B,C,D},'d',i1,'maxiters',10000,'epsilon',0.0001);
tt1=toc;
tic
 load('M.mat')
 load('U1.mat')
 load('U2.mat')
 load('U3.mat')
 load('U4.mat')
   
    rho=7;tau=6;
    lambda1=0.35;lambda2=0.3;lambda3=0.45;lambda4=0.35;
    d1=Dim;d2=Dim;d3=Dim;d4=Dim;d=Dim;kmax=option.maxiter;
    A_0=tensor(ones(d1,d2,d3,d4));
    B1_0 =zeros(d1,d);  
    B2_0 =zeros(d2,d);
    B3_0 =zeros(d3,d);
    B4_0 =zeros(d4,d);
    P_0=tensor(zeros(d,d,d,d));
    hatM_0=tensor(zeros(d1,d2,d3,d4));
    U1_0 =U1;U2_0 =U2;U3_0 =U3;U4_0 =U4;
    V1_0 =U1;V2_0 =U2;V3_0 =U3;V4_0 =U4;
 
 for k =1:kmax
%%    P_k  
P_k=ttm(M,{U1_0,U2_0,U3_0,U4_0},[1,2,3,4]);
%%    Vp_k
V_11=kron(kron(V4_0,V3_0),V2_0);
V_22=kron(kron(V4_0,V3_0),V1_0);
V_33=kron(kron(V4_0,V2_0),V1_0);
V_44=kron(kron(V3_0,V2_0),V1_0);
tg1=-B1_0+rho*(V1_0-U1_0);
t1=tenmat(P_k,1);
t2=tenmat(P_k,2);
t3=tenmat(P_k,3);
t4=tenmat(P_k,4);
tff1=-2*tenmat(M,1)*V_11*t1'+V1_0*t1*V_11.'*V_11*t1';
tf1=double(tff1);
F1=tg1+tf1;

tg2=-B2_0+rho*(V2_0-U2_0);
tff2=-2*tenmat(M,2)*V_22*t2'+V2_0*t2*V_22.'*V_22*t2';
tf2=double(tff2);
F2=tg2+tf2;

tg3=-B3_0+rho*(V3_0-U3_0);
tff3=-2*tenmat(M,3)*V_33*t3'+V3_0*t3*V_33.'*V_33*t3';
tf3=double(tff3);
F3=tg3+tf3;

tg4=-B4_0+rho*(V4_0-U4_0);
tff4=-2*tenmat(M,4)*V_44*t4'+V4_0*t4*V_44.'*V_44*t4';
tf4=double(tff4);
F4=tg4+tf4;
%prox
   V1_k = zeros (d1,d);
    for m = 1:i1
        if norm(F1(:,m)) >= 1/lambda1
           V1_k(:,m) = V1_0(:,m) - lambda1*(V1_0(:,m)/norm(V1_0(:,m)+norm(rand(2))));
        else
           V1_k(:,m) = 0;
        end
    end  
     V2_k = zeros (d2,d);
    for m = 1:i1
        if norm(F2(:,m)) >= 1/lambda2
           V2_k(:,m) = V2_0(:,m) - lambda2*(V2_0(:,m)/norm(V2_0(:,m)+norm(rand(2))));
        else
           V2_k(:,m) = 0;
        end
    end   

     V3_k = zeros (d3,d);
    for m = 1:i1
        if norm(F3(:,m)) >= 1/lambda3
           V3_k(:,m) = V3_0(:,m) - lambda3*(V3_0(:,m)/norm(V3_0(:,m)+norm(rand(2))));
        else
           V3_k(:,m) = 0;
        end
    end
     V4_k = zeros (d3,d);
    for m = 1:i1
        if norm(F4(:,m)) >= 1/lambda4
           V4_k(:,m) = V4_0(:,m) - lambda4*(V4_0(:,m)/norm(V4_0(:,m))+norm(rand(2)));
        else
           V4_k(:,m) = 0;
        end
    end
%%    Up_k 
Q_1=double((tenmat(A_0,1)+rho*tenmat(M,1))*V_11*t1')+rho*(V1_0-B1_0);
[U,S,V]=svd(Q_1);
U=U(:,1:d);
U1_k=U*V.';
Q_2=double((tenmat(A_0,2)+rho*tenmat(M,2))*V_22*t2')+rho*(V2_0-B2_0);
[U,S,V]=svd(Q_2);
U=U(:,1:d);
U2_k=U*V.';
Q_3=double((tenmat(A_0,3)+rho*tenmat(M,3))*V_33*t3')+rho*(V3_0-B3_0);
[U,S,V]=svd(Q_3);
U=U(:,1:d);
U3_k=U*V.';
Q_4=double((tenmat(A_0,4)+rho*tenmat(M,4))*V_44*t4')+rho*(V4_0-B4_0);
[U,S,V]=svd(Q_4);
U=U(:,1:d);
U4_k=U*V.';
%%     u   w  s
  U={U1_k,U2_k,U3_k,U4_k};
 views={A,B,C,D};
 n_samples = size(views{1},1);  
 epsilon=0.1;
 variances = cell(size(views));
   for i=1:4
        variances{i} =  (double(views{i})'*double(views{i}))/n_samples;
        variances{i} = variances{i} +  epsilon*ones(size(variances{i}));
   end
 Z_TCCAOS = zeros(n_samples,d*2);                                  
    for i=1:4
       Z_TCCAOS(:,(1+(i-1)*d):(i*d)) = double(views{i})*(pinv(variances{i})^1/2)*U{i};   %z的列数，1：d，d+1,�?2d�?2d+1�?3d
    end 
%% termination check
err1=0.01;
M_0=ttm(tensor(P_0),{U1_0.',U2_0.',U3_0.',U4_0.'},[1,2,3,4]);
%% termination check
 h1=@(X) lambda1*sum(vecnorm(X,2,2));
 h2=@(X) lambda2*sum(vecnorm(X,2,2));
 h3=@(X) lambda3*sum(vecnorm(X,2,2));
 h4=@(X) lambda4*sum(vecnorm(X,2,2));
err1=0.01;
pmu=frob(double(ttm(tensor(P_0),{U1_0,U2_0,U3_0,U4_0},[1,2,3,4]))-double(M));
pp1=h1(U1_0)+h2(U2_0)+h3(U3_0)+h4(U4_0);
pmu1=pmu+pp1;
%P_k=ttm(M,{U1_0,U2_0,U3_0,U4_0},[1,2,3,4]);
pu=frob(double(ttm(tensor(P_k),{U1_k,U2_k,U3_k,U4_k},[1,2,3,4]))-double(M));
pp2=h1(U1_k)+h2(U2_k)+h3(U3_k)+h4(U4_k);
pu1=pu+pp2;
if  norm(pu1-pmu1)<err1
        fprintf('break');
    break;
end
if k==kmax
 fprintf('iteration reaches maximum.');
end
 end
    t_TCCAOS=toc;
    t_TCCAOS=t_TCCAOS+tt1;

    [accr_TCCAOS, f1score_TCCAOS,~] = knnacc4(testRatio,train_num,test_num,Z_TCCAOS,label,class1);
    dataformat_TCCAOS = '%d-th experiment:  accr_TCCAOS = %f, f1score_TCCAOS = %f,time_TCCAOS=%f\n';
    dataValue_TCCAOS = [iExperiment, accr_TCCAOS,f1score_TCCAOS, t_TCCAOS];
    results_TCCAOS(iExperiment, :)=dataValue_TCCAOS;
    fprintf(dataformat_TCCAOS, dataValue_TCCAOS);
 
   end


   dataValue_TCCAO=mean(results_TCCAO, 1);
   STD_TCCAO=std(results_TCCAO(:,2));
    STD_TCCAOf1=std(results_TCCAO(:,3));
   fprintf('\nAverage:  Dimension =%d:  accr_TCCAO = %f, f1score_TCCAO = %f, time_TCCAO=%f,stdacc_TCCAO=%f,stdf1_TCCAL=%f\n',  Dim,dataValue_TCCAO(2:end),STD_TCCAO,STD_TCCAOf1);
    
   dataValue_TCCAOS=mean(results_TCCAOS, 1);
   STD_TCCAOS=std(results_TCCAOS(:,2));
    STD_TCCAOSf1=std(results_TCCAOS(:,3));
   fprintf('\nAverage: Dimension =%d:  accr_TCCAOS = %f,f1score_TCCAOS = %f, time_TCCAOS=%f,stdacc_TCCAOS=%f,stdf1_TCCAOS=%f\n',  Dim,dataValue_TCCAOS(2:end),STD_TCCAOS,STD_TCCAOSf1); 
end


