
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
    params = inputParser;    % 创建类，省略了（）
    params.addRequired('views',@iscell);   	% 放入 Results 中名称为 views,views的数据格式为cell
    params.addParameter('d',20,@(x) isscalar(x) & x > 0);   % 添加d默认参数，d的数据格式为标量
    params.addParameter('epsilon',0.01,@(x) isscalar(x) & x > 0);   % 添加epsilon默认参数，epsilon的数据格式为标量
    params.addParameter('maxiters', 50,@(x) isscalar(x) & x > 0);   % 添加maxiters默认参数，maxiters的数据格式为标量
    params.addParameter('verbosity', 1,@(x) isscalar(x));           % 添加verbosity默认参数，verbosity的数据格式为标量
    
    params.parse(views,varargin{:});    % varargin 必须解出来，调用者调用该函数时根据需要来改变输入参数的个数，以cell形式组成

    views = params.Results.views;
    d = params.Results.d;
    epsilon = params.Results.epsilon;
    maxiters = params.Results.maxiters;
    verbosity = params.Results.verbosity;
    
    
    %All views are assumed to contain equal number of samples
    n_samples = size(views{1},1);     %cell行数
    n_views = length(views);          %cell列数
    
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
    
    f = tucker_als(M,d,'maxiters',maxiters,'printitn',verbosity);   %cp分解
   U=f.U{i}
    Z = zeros(n_samples,d*2);       %z的列数？应该为m                            
    for i=1:n_views
        Z(:,(1+(i-1)*d):(i*d)) = double(views{i})*(pinv(variances{i})^1/2)*f.U{i};   %z的列数，1：d，d+1,：2d，2d+1：3d
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
%caxis([-1*colorValue 1*colorValue]);   %画频谱图
%colorbar;
    save tccadata2.txt -ascii Z         %保存z
end