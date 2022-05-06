%% Discription : 
  % This code is for the paper 'Fast Variational Bayesian Inference for
  % Temporally Correlated Sparse Signal Recovery'. 

  
  % The proposed algorithm can achieve higher performance and more
  % robustness with line 88;
  
  % Otherwise please normalize 'B' (line 98-100).

  %  Written by Zheng Cao

  
  function [X]=correlate_VBI(Y,A,K)


[M,T]=size(Y);
[~,P]=size(A);
F=A;


%% initialization
a=10^(-10);
b=a;
maxiter=300;
tol=1e-4;
alpha=1;
delta_inv=ones(P,1)*1;
[~,~,V]=svd(Y);
B=V(:,1:K)';
mu=zeros(P,K);
keep_list = [1:P]';
iter = 0;
Phi=F;

converged = false;
while ~converged
    
    iter = iter + 1;
   
 %% calculate mu and Sigma
 Phi_delta = Phi *  diag(delta_inv);
 %   Exx=[]; 
 for k= 1:K
     B_k=B(k,:);
     V_temp= 1/(alpha*(B_k*B_k'))*eye(M) + Phi_delta * Phi';
     Sigma=diag(delta_inv) -Phi_delta' * (V_temp \Phi_delta);
     temp=zeros(length(delta_inv),1);
     for i=1:K
         if i~=k
             temp=temp+mu(:,i)*B(i,:)*B_k';
         end
     end
     W1=Y*B_k'-Phi*temp;
     mu(:,k)=alpha*Sigma*Phi'*W1;
     Exx(:,k)= mu(:,k).*conj(mu(:,k))+ real(diag(Sigma ));
 end
    
        
        
  %%  update alpha
  resid=Y-Phi*mu*B;
  %   traceAGA1=[];traceAGA=[]; 
  for j=1:K
      PGP=diag(Phi*diag(delta_inv)*Phi');
      bkbk=B(j,:)*B(j,:)';
      traceAGA1(:,j)=sum( PGP./(  1 +  alpha*bkbk *PGP  )   );
      traceAGA(:,j)= traceAGA1(:,j)  * bkbk ;
  end
  b_k=b+ 0.5* ( norm(resid,'fro')^2 +  sum(traceAGA,2)  );
  a_k=a+ (M*T)/2;
  alpha=a_k/b_k;
    
    
    
  %% update delta
  delta_last=delta_inv;
  sum_temp1=sum(Exx,2);
  c_k=K/2+a;
  d_k=b+0.5*sum_temp1;
  delta_inv=d_k ./ c_k;
    

  %%  update B
  %   SA=[];
  for kk=1:K
      SA(:,kk)=Phi*mu(:,kk);
      p1=  SA(:,kk)'*SA(:,kk) +traceAGA(:,kk);
      temp11=zeros(length(delta_inv),1);
      for ii=1:K
          if ii~=kk
              temp11=temp11+mu(:,ii)*B(ii,:);
          end
      end
      W2=Y-Phi*temp11;
      p2=SA(:,kk)'*W2;
      B(kk,:) = p2/ p1;
   %  if norm(B(kk,:))<1
   %  B(kk,:)=B(kk,:)/norm(B(kk,:));
   %  end
  end

   
   %%  Set threshold and prune out the Values less than 10 ^ (- 3) in Delta_inv
  
%  if iter>0 
%     if min(delta_inv) < 10^(-3) % &  length(delta_inv)>= M
%         index=find( delta_inv>10^(-3) );
%         delta_inv= delta_inv(index);
%         delta_last=delta_last(index);
%         Sigma = Sigma(index,index,:);
%         Phi=Phi(:,index);
%         mu=mu(index,:);
%         Exx=Exx(index,:);
%         keep_list = keep_list(index);
%     end
%  end
    
    erro=  max(max(abs(delta_inv - delta_last)));  
    if erro < tol || iter >= maxiter
        converged = true;
    end
    
end

mu_est = zeros(P,K);
mu_est(keep_list,:) = mu;
X=mu_est*B;

