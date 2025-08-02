
% Syntax:  [reduction] = mvdr_tcca(views,d,epsilon)
%
% Inputs:
%    views - Cell array of NxM matrices
%    d - Final dimensionality
%    epsilon - Regularization trade-off factor, non-negative.
%    maxiters - Maximum number of ALS iterations in CP decomposition
%    verbosity - Fit error verbosity. If zero no info is printed, 
%                otherwise info is printed every n iterations.
%
% Outputs:
%    reduction - Multi-view dimensionality reduction
%    f - Factorization as ktensor
%
% Example: 
%    mvdr_ttcca({first_view, second_view})
%


function [Z,f] = mvdr_ttcca(views,varargin)
    %Parse inputs
    params = inputParser;    % �����࣬ʡ���ˣ���
    params.addRequired('views',@iscell);   	% ���� Results ������Ϊ views,views�����ݸ�ʽΪcell
    params.addParameter('d',20,@(x) isscalar(x) & x > 0);   % ���dĬ�ϲ�����d�����ݸ�ʽΪ����
    params.addParameter('epsilon',0.01,@(x) isscalar(x) & x > 0);   % ���epsilonĬ�ϲ�����epsilon�����ݸ�ʽΪ����
    params.addParameter('maxiters', 50,@(x) isscalar(x) & x > 0);   % ���maxitersĬ�ϲ�����maxiters�����ݸ�ʽΪ����
    params.addParameter('verbosity', 1,@(x) isscalar(x));           % ���verbosityĬ�ϲ�����verbosity�����ݸ�ʽΪ����
    
    params.parse(views,varargin{:});    % varargin ���������������ߵ��øú���ʱ������Ҫ���ı���������ĸ�������cell��ʽ���

    views = params.Results.views;
    d = params.Results.d;
    epsilon = params.Results.epsilon;
    maxiters = params.Results.maxiters;
    verbosity = params.Results.verbosity;
    
    
    %All views are assumed to contain equal number of samples
    n_samples = size(views{1},1);     %cell����
    n_views = length(views);          %cell����
    
    %Center each view
    for i=1:n_views
        views{i} = tensor(views{i} - repmat(mean(views{i} ), n_samples,1));   
    end
    
    %Calculate variances
    variances = cell(size(views));
    for i=1:n_views
        variances{i} =  (double(views{i})'*double(views{i}))/n_samples;
        variances{i} = variances{i} +  epsilon*ones(size(variances{i}));
    end
    %Calculate covariances
    covariances = [];
    for i=1:n_samples
       outer_product = views{1}(i,:);
       for j=2:length(views)
           outer_product = ttt(outer_product,views{j}(i,:));
       end
       if isempty(covariances)
           covariances = outer_product;
       else
           covariances = covariances+outer_product;
       end
    end   
    covariances = covariances / n_samples;
    
    M = covariances;
    for i=1:length(variances)
       M = ttm(M,pinv(variances{i})^1/2,i);
    end
    
    save M M;
    
    f = tucker_als(M,d,'maxiters',maxiters,'printitn',verbosity);   %cp�ֽ�
   U=f.U{i}
    Z = zeros(n_samples,d*2);       %z��������Ӧ��Ϊm                            
    for i=1:n_views
        Z(:,(1+(i-1)*d):(i*d)) = double(views{i})*(pinv(variances{i})^1/2)*f.U{i};   %z��������1��d��d+1,��2d��2d+1��3d
    end
    core=f.core
    U1=f.U{1};
    U2=f.U{2};
    U3=f.U{3};
    U4=f.U{4};
    save U1 U1;
    save U2 U2;
    save U3 U3;
    save U4 U4;
%caxis([-1*colorValue 1*colorValue]);   %��Ƶ��ͼ
%colorbar;
    save tccadata2.txt -ascii Z         %����z
end