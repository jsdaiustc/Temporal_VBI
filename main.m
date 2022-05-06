
clear
N = 40;                  % Row number of the dictionary matrix 
M = N * 5;               % Column number of the dictionary matrix
L = 8;                   % Column Number of signal vectors
K = 10;                  % Number of nonzero rows (i.e. source number) in the solution matrix
beta = ones(K,1)*0.9;    % Temporal correlation of each nonzero row in the solution matrix.
SNR=30;



%% processes have different AR coefficients
 A = randn(N,M);
 nonzeroW(:,1) = randn(K,1);
 for i = 2 : L*100
      nonzeroW(:,i) = beta .* nonzeroW(:,i-1) + sqrt(1-beta.^2).*(ones(K,1).*randn(K,1));
 end
 nonzeroW = nonzeroW(:,end-L+1:end);   % Ensure the AR processes are stable
% Normalize each row
nonzeroW = nonzeroW./( sqrt(sum(nonzeroW.^2,2)) * ones(1,L) );

% Rescale each row such that the squared row-norm distributes in [1,scalefactor]
scalefactor = 3; 
mag = rand(1,K); mag = mag - min(mag);
mag = mag/(max(mag))*(scalefactor-1) + 1;
nonzeroW = diag(sqrt(mag)) * nonzeroW;

% Locations of nonzero rows are randomly chosen
ind = randperm(M);
indice = ind(1:K);
S = zeros(M,L);
S(indice,:) = nonzeroW;

signal=A*S;
stdnoise = std(reshape(signal,N*L,1))*10^(-SNR/20);
noise = randn(N,L) * stdnoise;
Y=signal+noise;


%%
Weight=correlate_VBI(Y,A,L);
mse_VBI=(norm(S - Weight,'fro')/norm(S,'fro'))^2;
figure
subplot(1,2,1)
plot(S);
title('true')
subplot(1,2,2)
plot(Weight);
title('estimated')
