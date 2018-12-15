function K = cnn_kron(A,B)
%CNN_KRON Kronecker tensor product.
%   Modified from KRON, this version support a 3D input A tensor
[ma,na,la] = size(A);
[mb,nb] = size(B);

A = reshape(A,[1 ma 1 na la]);
B = reshape(B,[mb 1 nb 1]);
K = reshape(bsxfun(@times,A,B),[ma*mb na*nb la]);

end

