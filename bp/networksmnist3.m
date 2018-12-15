% A BP network for MNIST data
% 隐含层ReLU激活函数+均方误差代价函数+L2规则化

% Author: Guo chengfeng
% Version 1.1
% date: 2018-3-19


%神经网络的层次
ntrain = length(y_train);
arch = [784,100,100,10];
nlayer = length(arch);
mini_batch_size = 100;
max_iteration = 50000;
eta = 1;
lambda = 5;
% mini = 20, eta = 6 eps= 5000. 可以达到0.946
%神经网络参数初始化
weight = cell(1,nlayer);
bias = cell(1,nlayer);
nabla_weight = cell(1,nlayer);
nabla_bias = cell(1,nlayer);
a = cell(1,nlayer);
z = cell(1,nlayer);
rstep = 100;
for in = 2:nlayer
    weight{in} = rand(arch(in),arch(in-1))-0.5;
    bias{in} = rand(arch(in),1)-0.5;
    nabla_weight{in} = rand(arch(in),arch(in-1));
    nabla_bias{in} = rand(arch(in),1);
end
for in = 1:nlayer
    a{in} = zeros(arch(in),mini_batch_size);
    z{in} = zeros(arch(in),mini_batch_size);
end
accuracy = zeros(1,ceil(max_iteration/rstep));

iaa = 0;
for ip = 1:max_iteration
    
    pos = randi(ntrain-mini_batch_size);
    x = x_train(:,pos+1:pos+mini_batch_size);
    y = y_train(:,pos+1:pos+mini_batch_size);
    %hidden: sigmoid, output: sigmoid, cost: 'quadratic'
    a{1} = x;
    [a,z]=bp_feedforward(@acti_relu,@acti_sigmoid,weight,bias,nlayer,a,z);
    [weight,bias] = bp_backpropagation(@acti_relu_prime,@acti_sigmoid_prime,'quadratic',weight,bias,...
        nabla_weight,nabla_bias,nlayer,mini_batch_size,eta,a,z,y,lambda,ntrain);
    if mod(ip,rstep) == 0
        iaa = iaa+1;
        accuracy(iaa) = evaluatemnist(@acti_relu,@acti_sigmoid,x_valid,y_valid,weight,bias,nlayer);
        plot(accuracy);
        title(['Accuracy:',num2str(accuracy(iaa))]);
        getframe;
    end
end
figure
plot(accuracy,'r','linewidth',2);
ylim([0.95,0.99])
legend(num2str(arch))
title(['Sig+Sig+Quad: ',num2str(max(accuracy)*100),'%'])
grid on
accuracy_test = evaluatemnist(@acti_relu,@acti_sigmoid,x_test,y_test,weight,bias,nlayer);
disp(['test_data: ',num2str(accuracy_test*100),'%']);
