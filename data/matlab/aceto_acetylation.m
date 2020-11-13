close all;
clear variables;

% Identification on/off
identification = true;
number_repetitions = 100;

% Size of database (1 to 3)
database_size = 1;

% Number of exported points
npoints = 151;

% Noise level
noise_level = 0.01;

% Number of species/reactions
ns = 5;
nr = 3;

% Stoichiometric matrix
nu  = [ -1 -1 1 0 0; ...
         0 -2 0 1 0; ...
         0 -1 0 0 1 ];

% Reactor data
Cin = [0 6 0 0 0]';                 % inlet concentration (mol/l)
V   = 1;                            % reactor volume (l)
C0  = [0.30 0.14 0.08 0.01 0]';     % initial concentration (mol/l)
C0  = [1.00 1.00 1.00 1.00 1.00]';  % initial concentration (mol/l)
kappa = [0.053 0.128 0.028]';       % kinetic constants (mol,l,min)
tf  = 20;                           % total residence time (min)

% Reference values
if (database_size == 1)
    Qin = [0.3]; % volumetric flow rate (l/min)
    Cc =  [0.5]; % catalyst concentration (mol/l)
elseif (database_size == 2)
    Qin = [0.1:0.025:0.5]; % volumetric flow rate (l/min)
    Cc  = [0.5];           % catalyst concentration (mol/l)
elseif (database_size == 3)    
    Qin = [0.1:0.025:0.5]; % volumetric flow rate (l/min)
    Cc  = [0.4:0.05:0.6];  % catalyst concentration (mol/l)
end

%% Database calculations

t_span = 0:tf/(npoints-1):tf;

% Memory allocation for input/output variables
Y_overall = zeros(npoints*length(Qin)*length(Cc),ns);
X_overall = zeros(npoints*length(Qin)*length(Cc),3);

% Database construction
count = 0;
for i=1:length(Cc)
    for j=1:length(Qin)
   
        options = odeset('AbsTol',1e-9, 'RelTol',1e-6);
        [t, C]=ode45(@reactor_equations, t_span, C0, options, ...
                     nu,kappa,Cc(i),Cin,Qin(j),V); 
        
        for kk=1:npoints
            X_overall(count+kk,:) = [t(kk), Qin(j), Cc(i)];
            Y_overall(count+kk,:) = C(kk,:);
        end
        
        count = count+npoints;

    end
end

% Write on file
x1 = t;
x2 = Qin;
x3 = Cc;
save aceto_acetylation.mat x1 x2 x3 X_overall Y_overall


%% Plot reference curves

% ODE solution
options = odeset('AbsTol',1e-9, 'RelTol',1e-6);
[t, C]=ode45(@reactor_equations, t_span, C0, options, ...
                 nu,kappa,Cc(1),Cin,Qin(1),V);

% Noisy profiles
sigmas = noise_level*max(C);
delta_basis = randn(npoints,ns); 
delta(:,1:ns) = delta_basis(:,1:ns) .* sigmas;
Cnoisy = C + delta;

% Figure
figure; hold on;
plot(t_span, C(:,1),'r-');
plot(t_span, C(:,2),'b-');
plot(t_span, C(:,3),'g-');
plot(t_span, C(:,4),'y-' );
plot(t_span, C(:,5),'k-');
plot(t_span,Cnoisy(:,1),'ro' );
plot(t_span,Cnoisy(:,2),'bo' );
plot(t_span,Cnoisy(:,3),'go' );
plot(t_span,Cnoisy(:,4),'yo' );
plot(t_span,Cnoisy(:,5),'ko' );
xlabel('time (min)'); ylabel('concentration (mol/l)');
legend('A', 'B', 'C', 'D', 'E');
hold off; 

%% Identification
if (identification == true)

    kappa_true = kappa;
    
    for kk = 1:number_repetitions
        
        kappa_min = [0.01, 0.01, 0.01];
        kappa_max = [1.00, 1.00, 1.00];
        kappa0 = [0.10, 0.10, 0.10];

        kappa(:,kk) = lsqnonlin(@obj_func, kappa0, kappa_min, kappa_max, [], ...
                      t_span,C0,nu,Cc(1),Cin,Qin(1),V,Cnoisy);
    end
    
    kappa = kappa';
    kappa_mean = mean(kappa);
    err_rel = abs(kappa_mean-kappa_true')./kappa_true * 100.;
    fprintf('kappa = %e %e %e \n', kappa_mean(1), kappa_mean(2), kappa_mean(3));
    fprintf('error = %e %e %e \n', err_rel(1), err_rel(2), err_rel(3));
    
end

%% ODE system
function dCdt = reactor_equations(~,C, nu,kappa,Cc,Cin,Qin,V)

    % Reaction rates
    r(1) = kappa(1)*C(1)*C(2)*Cc;
    r(2) = kappa(2)*C(2)^2*Cc;
    r(3) = kappa(3)*C(2);

    % Formation rates
    R = nu' * r';
    
    % Residence time
    tau = V/Qin;
    
    % Equations
    dCdt = (Cin-C)/tau + R;

end

function f = obj_func(x, t_span,C0,nu,Ccat,Cin,Qin,V,Cnoisy)

    kappa = x;
    options = odeset('AbsTol',1e-9, 'RelTol',1e-6);
    [~, C]=ode45(@reactor_equations, t_span, C0, options, ...
                     nu,kappa,Ccat,Cin,Qin,V);
                 
    f1 = C(:,1)-Cnoisy(:,1);
    f2 = C(:,2)-Cnoisy(:,2);
    f3 = C(:,3)-Cnoisy(:,3);
    f4 = C(:,4)-Cnoisy(:,4);
    f5 = C(:,5)-Cnoisy(:,5);
    
    f = [f1;f2;f3;f4;f5];
    
end

