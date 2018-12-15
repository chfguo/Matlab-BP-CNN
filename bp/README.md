%第一步运行 loadMnistDataScript 加载数据
%第一个版本
%运行 networksmnist 即可

%第二个版本（识别率高）
%运行 networksmnist2 即可

%第二个版本
%测试结果，以下参数最高可达到98.34%的识别率
% 结果见./fig/a80-80-m-100-me-50000-e-1-l-5.png
% arch = [784,80,80,10];
% mini_batch_size = 100;
% max_max_iteration = 50000;
% eta = 1;
% lambda = 5;

%测试结果，以下参数最高可达到98.39%的识别率
% 结果见./fig/a100-100-m-100-me-50000-e-1-l-5.png
% arch = [784,100,100,10];
% mini_batch_size = 100;
% max_max_iteration = 50000;
% eta = 1;
% lambda = 5;
loadMnistDataScript;
networksmnist2;
