function cnn = cnn_initialize(cnn)
%CNN_INIT initialize the weights and biases, and other parameters
%   
index = 0;
num_layer = numel(cnn.layer);
for in = 1:num_layer
    switch cnn.layer{in}{1}
        case 'input'
            index = index + 1;
            height = cnn.layer{in}{3}(1);
            width = cnn.layer{in}{3}(2);
            mini_size = cnn.layer{in}{2};
            cnn.weights{index} = [];
            cnn.biases{index} = [];
            cnn.nabla_w{index} = [];
            cnn.nabla_b{index} = [];
            %n*n*m
            cnn.a{index} = [];
            cnn.z{index} = [];
            cnn.delta{index} = [];
            cnn.mini_size = mini_size;
        case 'conv'
            index = index + 1;
            %kernel height, width, number
            ker_height = cnn.layer{in}{3}(1);
            ker_width = cnn.layer{in}{3}(2);
            ker_num = cnn.layer{in}{2};
            cnn.weights{index} = grand(ker_height,ker_width,ker_num) - 0.5;
            cnn.biases{index} = grand(1,ker_num) - 0.5;
            cnn.nabla_w{index} = zeros(ker_height,ker_width,ker_num);
            cnn.nabla_b{index} = zeros(1,ker_num);
            height = height - ker_height + 1;
            width = width - ker_width + 1;
            cnn.a{index} = zeros(height,width,mini_size,ker_num);
            cnn.z{index} = zeros(height,width,mini_size,ker_num);
            cnn.delta{index} = zeros(height,width,mini_size,ker_num);
        case 'pool'
            index = index + 1;
            %kernel height, width, number
            ker_height = cnn.layer{in}{3}(1);
            ker_width = cnn.layer{in}{3}(2);
            cnn.weights{index} = [];
            cnn.biases{index} = [];
            cnn.nabla_w{index} = [];
            cnn.nabla_b{index} = [];
            height = height / ker_height;
            width = width / ker_width;
            cnn.a{index} = zeros(height,width,mini_size,ker_num);
            cnn.z{index} = [];
            cnn.delta{index} = zeros(height,width,mini_size,ker_num);
        case 'flat'
            index = index + 1;
            cnn.weights{index} = [];
            cnn.biases{index} = [];
            cnn.nabla_w{index} = [];
            cnn.nabla_b{index} = [];

            cnn.a{index} = zeros(height*width*ker_num,mini_size);
            cnn.z{index} = [];
            cnn.delta{index} = zeros(height*width*ker_num,mini_size);
        case 'full'
            index = index + 1;
            %kernel height, width, number
            neuron_num = cnn.layer{in}{2};
            neuron_num0 = size(cnn.a{in-1},1);
            
            cnn.weights{index} = grand(neuron_num,neuron_num0) - 0.5;
            cnn.biases{index} = grand(neuron_num,1) - 0.5;
            cnn.nabla_w{index} = zeros(neuron_num,neuron_num0);
            cnn.nabla_b{index} = zeros(neuron_num,1);
    
            cnn.a{index} = zeros(neuron_num,mini_size);
            cnn.z{index} = zeros(neuron_num,mini_size);
            cnn.delta{index} = zeros(neuron_num,mini_size);
            
        case 'output'
             index = index + 1;
            %kernel height, width, number
            neuron_num = cnn.layer{in}{2};
            neuron_num0 = size(cnn.a{in-1},1);
            
            cnn.weights{index} = grand(neuron_num,neuron_num0) - 0.5;
            cnn.biases{index} = grand(neuron_num,1);
            cnn.nabla_w{index} = zeros(neuron_num,neuron_num0);
            cnn.nabla_b{index} = zeros(neuron_num,1);
    
            cnn.a{index} = zeros(neuron_num,mini_size);
            cnn.z{index} = zeros(neuron_num,mini_size);
            cnn.delta{index} = zeros(neuron_num,mini_size);
        otherwise
            
    end
end
end

