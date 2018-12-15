function cnn = cnn_backpropagation2(cnn,y)
%CNN_BP CNN backpropagation

num = numel(cnn.layer);

for in = num:-1:2

switch cnn.layer{in}{1}
    case 'conv'
        
        ker_h = cnn.layer{in+1}{3}(1);
        ker_w = cnn.layer{in+1}{3}(2);
        kernel = ones(ker_h,ker_w)/ker_h/ker_w;
      
        [~,~,mini_size,kernel_num] = size(cnn.delta{in+1});
        cnn.nabla_w{in}(:) = 0;
        cnn.nabla_b{in}(:) = 0;
        for ik = 1:kernel_num
             cnn.delta{in}(:,:,:,ik) = cnn_kron(cnn.delta{in+1}(:,:,:,ik),kernel).*relu_prime(cnn.z{in}(:,:,:,ik));
             for im = 1:mini_size
                cnn.nabla_w{in}(:,:,ik) = cnn.nabla_w{in}(:,:,ik) +...
                    conv2(rot90(cnn.a{in-1}(:,:,im),2),cnn.delta{in}(:,:,im,ik),'valid');
             end
            cnn.nabla_b{in}(ik) = mean(mean(mean(cnn.delta{in}(:,:,:,ik))));
            cnn.nabla_w{in}(:,:,ik) = cnn.nabla_w{in}(:,:,ik)/mini_size;
        end
        
%          for ik = 1:kernel_num
%             for im = 1:mini_size
%                 cnn.delta{in}(:,:,im,ik) = kron(cnn.delta{in+1}(:,:,im,ik),kernel).*relu_prime(cnn.z{in}(:,:,im,ik));
%                 cnn.nabla_w{in}(:,:,ik) = cnn.nabla_w{in}(:,:,ik) +...
%                     conv2(rot90(cnn.a{in-1}(:,:,im),2),cnn.delta{in}(:,:,im,ik),'valid');
%                 cnn.nabla_b{in}(ik) = cnn.nabla_b{in}(ik) + mean(mean(cnn.delta{in}(:,:,im,ik)));
%             end
%             cnn.nabla_w{in}(:,:,ik) = cnn.nabla_w{in}(:,:,ik)/mini_size;
%             cnn.nabla_b{in}(ik) = cnn.nabla_b{in}(ik)/mini_size;
%         end
        
    case 'pool'
        [height,width,mini_size,kernel_num] = size(cnn.a{in});
        for ik = 1:mini_size
            cnn.delta{in}(:,:,ik,:) = reshape(cnn.delta{in+1}(:,ik),[height,width,kernel_num]);
        end
    case 'flat'
        cnn.delta{in} = cnn.weights{in+1}'*cnn.delta{in+1};
    case 'full'
        cnn.delta{in}= cnn.weights{in+1}'*cnn.delta{in+1}.*sigmoid_prime(cnn.z{in});
        cnn.nabla_w{in} = cnn.delta{in}*(cnn.a{in-1})'/cnn.mini_size;
        cnn.nabla_b{in} = mean(cnn.delta{in},2);
    case 'output'
        cnn.delta{in}= (cnn.a{in} - y);
        cnn.nabla_w{in} = cnn.delta{in}*(cnn.a{in-1})'/cnn.mini_size;
        cnn.nabla_b{in} = mean(cnn.delta{in},2);
    otherwise
        
end

end

eta = cnn.eta;
lambda = cnn.lambda;
ntrain = cnn.ntrain;
% update models
for in = 1:num
    cnn.weights{in} = (1-eta*lambda/ntrain)*cnn.weights{in} - eta*cnn.nabla_w{in};
    cnn.biases{in} = (1-eta*lambda/ntrain)*cnn.biases{in} - eta*cnn.nabla_b{in};
end

end

