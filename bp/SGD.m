function [weight,bias] = SGD(hidenActiFcn,outputActiFcn,weight,bias,nabla_weight,nabla_bias,nlayer,mini_batch_size,eta,a,z,y,lambda,n)
%SGD stochastic gradient descent

delta = (a{nlayer}-y).*outputActiFcn(z{nlayer});
nabla_bias{end} = mean(delta,2);
nabla_weight{end} = (delta*a{end-1}')/mini_batch_size;

for in = nlayer-1:-1:2
    delta = weight{in+1}'*delta.*hidenActiFcn(z{in});
    nabla_bias{in} = mean(delta,2);
    nabla_weight{in} = (delta*a{in-1}')/mini_batch_size;
end
for in = 2:nlayer
   weight{in} =  (1-eta*lambda/n)*weight{in} - eta*nabla_weight{in};
   bias{in} = bias{in} - eta*nabla_bias{in};
end


end

