% clear;
p = 100; %% dimension
T = 1000; %% sample size
c = 2; %% condition number of the low rank component
N_tot = 100; %% number of replicates
r = 2; %% latent rank
delta = 1; %% magnitude of residual covariances VS their theoretical maximum
deltabis = 0.7; %% threshold to set surviving residual elements
tau = 3; %% scale parameter VS p
alpha = 0.9; %% latent variance parameter
imm = 1i; %% imaginary unit

same_rank = 1; %% 0 is wrong. Set to 1. can be removed
sparse_s = 2; %%  can be removed 0 for fixed residual spectral matrix across frequencies
sparse_diff = 0; %% 1 to change the residual pattern across coefficient matrices
coef_lag = [0 1 0]; %% squared lag coefficients
new_method = 1; %% spectral computation method
past_method = 0; %% can be removed
% not 
k_1 = -0.2; %% var parameter
var_1 = 0; %% if var1 desired 
var_2 = 0; %% if var2 desired
%
no_pert = 1; %% 0 to perturb latent eigenvalues / for user to choose
pert_soft = 0; %% 1 for soft perturbation, 0 for extreme perturbation / fix to 1 
prec_pert = 0; %% parameter controlling the amount of perturbation /fix to .1
f = 0:1/12:0.5; %% sampled frequencies (include 0.5 for tecnical reasons)

n_lag = 0;

% define shape
shape = 'from high to low';

[E_r, Lambda] = generate_val_vec(p, r, c, tau, alpha);
delay = lowRank_coeff_generation(p, r, coef_lag, pert_soft, no_pert, E_r);

% low rank componant generation: eigenvectors & eigenvalues
function [E_r, Lambda] = generate_val_vec(p, r, c, tau, alpha)
    % low rank component generation: eigenvectors
    
    R = zeros(p,p);
    V = zeros(p,p);
    E = zeros(p,p);
    v = zeros(p,1);
    K = rand(p)*eye(p);
    E(:,1) = K(:,1);
    
    for j=2:p
        for i=1:(j-1)
            R(i,j) = dot(K(:,j),E(:,i))/(norm(E(:,i))^2); % slow; can be improved
        end

        for i=1:(j-1)
            V(:,i) = R(i,j) * E(:,i);
        end

        for h=1:p
            v(h) = sum(V(h,1:(j-1)));
        end
        
       E(:,j) = K(:,j) - v;
    end
    
    for i=1:p
        E(:,i) = E(:,i)/norm(E(:,i));
    end
    
    v = randperm(p,r);
    E_r = E(:,v); 
    
    % low rank component generation: eigenvectors
    lambda = zeros(r, length(c));
    Lambda = zeros(r, r, length(c));
    condn = zeros(1, length(c));
    normFrL = zeros(1, length(c));
    
    for i=1:length(c)
        a = [1-c(i) r-1];
        f_past = [(r-1)+c(i) sum(1:(r-2))];
        A = [a;f_past];
        B = [0 tau*alpha*p];
        sol(:,i) = linsolve(A, B');
        lambda(r, i) = sol(1, i);
        
        for q=2:r 
            lambda(r-q+1,i) = lambda(r,i) + ((q-1)) * sol(2,i);
        end
        
        Lambda(:,:,i) = diag(lambda(:,i));
        condn(i) = cond(Lambda(:,:,i),2);
    end

    %Lambda(2,2)=Lambda(2,2)+1/30;
    %Lambda(3,3)=Lambda(3,3)-1/30;

    for i=1:length(c)
        normFrL(i) = norm(Lambda(:,:,i),'fro');
    end

    % B_new=zeros(p);
    B = E_r * Lambda * E_r';
    
    %{
    rank(B)
    trace(B)
    diag_B=svds(B,rank(B))
    diag_B(rank(B))/diag_B(1)
    %}
end

%{
T_A=(n_lag+1)*p;

R=zeros(T_A,T_A);
V=zeros(T_A,T_A);
E=zeros(T_A,T_A);
v=zeros(T_A,1);
K=rand(T_A)*eye(T_A);
rank(K);
E(:,1)=K(:,1);
%for i=1:T_A
 %   for j=(i+1):T_A
%R(i,j)=dot(K(j,:),E(i,:))/norm(E(i,:));
%    end;
%end;
for j=2:T_A
    for i=1:(j-1)
        R(i,j)=dot(K(:,j),E(:,i))/(norm(E(:,i))^2);
    end;
    for i=1:(j-1)
        V(:,i)=R(i,j)*E(:,i);
    end;
    for h=1:T_A
        v(h)=sum(V(h,1:(j-1)));
    end;
   E(:,j)=K(:,j) - v;
end;

for i=1:T_A
    E(:,i)=E(:,i)/norm(E(:,i));
end;
rank(E)
E
E'*E

%v=randperm(p,r);
%v
E_r=E(:,1:(n_lag+1))
rank(E_r)

E_r_past=E_r
v=randperm(p,r);
v
E_r=E(:,v)
%}

function [AA_delay_all] = lowRank_coeff_generation(p, r, coef_lag, pert_soft, shape, no_pert, E_r)

    % var_1==0 var_2==0
   
    n_boh = 1;

    if same_rank==1
        n_lag = length(coef_lag) - 1;
        lag_0 = 0;
        clear A_delay Coef_lag

        for lag = 0:n_lag
            
            if lag < 2
                
                if shape == 'from high to low'
                    ones_perm = [-1 1 0];
                
                elseif shape == 'from low to high'
                    ones_perm = [1 -1 0];
                
                elseif shape == 'U shape reversed'
                    ones_perm=[-1 0 1];
                
                elseif shape == 'U shape'
                    ones_perm = [1 0 -1];
                    
                else
                    msg = 'Wrong shape';
                    error(msg)
                
                % lag coefficients for spectral shape:
                % da alto verso il basso (ones_perm=[-1 1 0]) o 
                % dal basso verso l'alto (ones_perm=[1 -1 0])
                % oppure (vedi 'spectral_generation_U_shape')
                % U rovesciata (ones_perm=[-1 0 1]) o
                % U (ones_perm=[1 0 -1])
                ones_2 = [ones(1) -ones(1)];
                ones_perm = ones_2(randperm(2));
                sign_2 = ones_perm(lag+1);
            end


            if lag==2
                ones_2(lag + 1) = 0;
                ones_perm(lag + 1) = 0;
                sign_2(lag + 1) = 0;
            end

            % low rank semi-coefficient matrices generation
            if no_pert==1
                A_delay(:,:,lag+1) = ones_perm(lag+1) * sqrt(coef_lag(lag+1)) * E_r(lag_0*p+1:(lag_0+1)*p,:);
            end

            % latent eigenvalue perturbation
            if no_pert==0

                % gamma: extreme    
                if pert_soft==0

                    vec_coef = repmat(r*(coef_lag(lag+1)), r, 1);
                    vec_lag = gamrnd(repmat(vec_coef, 1, n_boh),1);
                    vec_sum = sum(vec_lag);
                    %vec_lag = vec_lag./repmat(vec_sum, size(vec_lag, 1), 1);
                    vec_lag = vec_lag./(vec_sum)*(r*(coef_lag(lag+1)));

                    if lag < n_lag
                        Coef_lag(:,:,lag+1) = diag(vec_lag);
                    end

                    if lag == n_lag
                        for i=1:r
                            Coef_lag(i,i,lag+1) = sum((coef_lag)) - sum(Coef_lag(i,i,:));
                        end
                    end
                end

                % normal: soft
                if pert_soft == 1
                    if lag < n_lag
                        pert_coef = (-prec_pert*coef_lag(lag+1)+2*prec_pert*coef_lag(lag+1)*rand(1,r)); % creating perturbation coef
                        for i=1:r
                            Coef_lag(i,i,lag+1) = coef_lag(lag+1) + pert_coef(i); % sum perturbation at each lag
                        end
                    end

                    if lag == n_lag
                        for i = 1:r
                            Coef_lag(i,i,lag+1) = sum((coef_lag)) - sum(Coef_lag(i,i,:));
                        end
                    end
                end

                A_delay(:,:,lag+1) = E_r(lag_0*p+1:(lag_0+1)*p,:) * sqrt(Coef_lag(:,:,lag+1));
            end
        end
    end

    size(E_r)
    size(A_delay)

    for lag=0:n_lag
        for ncomp=1:r
            norm_comp(lag+1,ncomp) = norm(A_delay(:,ncomp,lag+1));
        end
    end

    for n_comp=1:r
        norm_r(n_comp) = norm(E_r(:, n_comp));
    end

    for lag_n=n_lag:-1:0
        for lag=0:lag_n
            u_lag = n_lag - lag_n;
            AA_delay_all(:,:,lag+1,u_lag+1) = A_delay(:,:,lag+u_lag+1) * Lambda(1:r,1:r) * A_delay(:,:,lag+1)';
        end
    end

    end
end