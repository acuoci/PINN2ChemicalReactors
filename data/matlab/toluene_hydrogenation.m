close all;
clear variables;

% Size of database (1 to 3)
database_size = 1;

% Number of exported points
npoints = 201;

% Number of species/reactions
ns = 3;
nr = 3;

% Reactor data
C0 = [1 0 0]';              % initial concentration (mol/l)
kH1 = 0.023; kD1 = 0.005;   % reaction 1 parameters
k2 = 0.011;                 % reaction 2 parameters
Krel = [1.9 1 1.8]';        % adsorption kinetic constants
tf = 400;                   % total residence time (min)

% Reference values
if (database_size == 1)
    v2 = [1];
    v3 = [1];
elseif (database_size == 2)
    v2 = [1];
    v3 = [1];
elseif (database_size == 3)    
    v2 = [1];
    v3 = [1];
end


%% Database calculations

t_span = 0:tf/(npoints-1):tf;

% Memory allocation for input/output variables
Y_overall = zeros(npoints*length(v2)*length(v3),ns);
X_overall = zeros(npoints*length(v2)*length(v3),3);

% Database construction
count = 0;
for i=1:length(v3)
    for j=1:length(v2)
   
        options = odeset('AbsTol',1e-9, 'RelTol',1e-6);
        [t,C]=ode45(@reactor_equations,t_span,C0, options, ...
                    Krel,kH1,kD1,k2);
        
        for k=1:npoints
            X_overall(count+k,:) = [t(k), v2(j), v3(i)];
            Y_overall(count+k,:) = C(k,:);
        end
        
        count = count+npoints;

    end
end

% Write on file
x1 = t;
x2 = v2;
x3 = v3;
save toluene_hydrogenation.mat x1 x2 x3 X_overall Y_overall


%% Plot reference curves

% ODE solution
options = odeset('AbsTol',1e-9, 'RelTol',1e-6);
[t,C]=ode45(@reactor_equations,t_span,C0, options, ...
            Krel,kH1,kD1,k2);

% Noisy profiles
sigmas = 0.01*max(C);
delta_basis = randn(npoints,ns); 
delta(:,1:ns) = delta_basis(:,1:ns) .* sigmas;
Cnoisy = C + delta;

% Figure
figure; hold on;
plot(t_span, C(:,1),'r-');
plot(t_span, C(:,2),'b-');
plot(t_span, C(:,3),'g-');
plot(t_span,Cnoisy(:,1),'ro' );
plot(t_span,Cnoisy(:,2),'bo' );
plot(t_span,Cnoisy(:,3),'go' );
xlabel('time (min)'); ylabel('concentration (mol/l)');
legend('A', 'B', 'C');
hold off; 


%% ODE system
function dCdt = reactor_equations(~,C, Krel,kH1,kD1,k2)

    % Surface coverages
    teta = Krel.*C / sum(Krel.*C);

    % Reaction rates
    rH1 = kH1*teta(1);
    rD1 = kD1*teta(2);
    r2  = k2*teta(2);
    
    % Equations
    dCdt(1) = -rH1+rD1;
    dCdt(2) =  rH1-rD1-r2;
    dCdt(3) =  r2;
    
    dCdt = dCdt';

end