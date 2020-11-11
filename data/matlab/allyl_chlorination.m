%% Allyl-chlorination

% P + Cl2 -> A + HCl r1=k1*Pp*Pcl2
% P + Cl2 -> D       r2=k2*Pp*Pcl2
%
% k1 = 9.02e-5*exp(-63267.20/RT) [mol/m3/s/Pa]
% k2 = 5.12e-9*exp(-15956.36/RT) [mol/m3/s/Pa]
%
% Order of species: Cl2, P, A, HCl, D

clear variables;
close all;

global A E ;
global Ai Vtot;
global Xin;

% Size of database (1 to 3)
database_size = 1;

% Number of exported points
npoints = 201;

% Number of species/reactions
ns = 5;
nr = 2;

% Reactor data
A = [9.02e-5, 5.12e-9];     % frequency factors [mol/m3/s/Pa]
E = [63267.20, 15956.36];   % activation energies [J/mol/K]
L = 8;                      % reactor length [m]
di = 0.05;                  % internal diameter [m]
teta = 4./1.;               % ratio FP/FCl2 (inlet flow rates)
P = 2.02e5;                 % pressure [Pa]

% Preprocessing
Ai = pi*di^2/4;                 % cross section area [m2]
Vtot = Ai*L;                    % total volume [m3]
Xin = [1. teta 0 0 0]/(1+teta); % inlet molde fractions [-]
   
% Reference values (isothermal conditions)
if (database_size == 1)
    T = [573];             % temperature [K]
    Ftot = [200]/3600;     % total molar flow rate [mol/s]
elseif (database_size == 2)
    T = [573 598 623 648 673];  % temperature [K]
    Ftot = [200]/3600;          % total molar flow rate [mol/s]
elseif (database_size == 3)    
    T = [573 598 623 648 673];       % temperature [K]
    Ftot = [200 250 300 350]/3600;   % total molar flow rate [mol/s]
end

%% Run complete set of isothermal simulations

V_span = 0:Vtot/(npoints-1):Vtot;

% Memory allocation for input/output variables
Y_overall = zeros(npoints*length(T)*length(Ftot),ns);
X_overall = zeros(npoints*length(T)*length(Ftot),3);

% Database construction
count = 0;
for i=1:length(Ftot)
    for j=1:length(T)
      
        [V,F, z,C] = RunIsothermal(P, T(j),Ftot(i), npoints);
        
        for k=1:npoints
            X_overall(count+k,:) = [V(k), T(j), Ftot(i)];
            Y_overall(count+k,:) = F(k,:);
        end

        count = count+npoints;

    end
end

% Write on file
x1 = V;
x2 = T;
x3 = Ftot;
save allyl_chlorination.mat x1 x2 x3 X_overall Y_overall

%% Plot reference curves

% ODE solution
[V,F, z,C] = RunIsothermal(P, T(1),Ftot(1), npoints);

% Noisy profiles
sigmas = 0.01*max(C);
delta_basis = randn(npoints,ns); 
delta(:,1:ns) = delta_basis(:,1:ns) .* sigmas;
Cnoisy = C + delta;

% Figure
figure; hold on;
plot(z, C(:,1),'r-');
plot(z, C(:,2),'b-');
plot(z, C(:,3),'g-');
plot(z, C(:,4),'y-' );
plot(z, C(:,5),'k-');
plot(z,Cnoisy(:,1),'ro' );
plot(z,Cnoisy(:,2),'bo' );
plot(z,Cnoisy(:,3),'go' );
plot(z,Cnoisy(:,4),'yo' );
plot(z,Cnoisy(:,5),'ko' );
xlabel('axial coordinate (m)'); ylabel('concentration (mol/m3)');
legend('Cl2', 'P', 'A', 'HCl', 'D');
hold off;

% Figure
figure; hold on;
plot(z, F(:,1),'r-');
plot(z, F(:,2),'b-');
plot(z, F(:,3),'g-');
plot(z, F(:,4),'y-' );
plot(z, F(:,5),'k-');
xlabel('axial coordinate (m)'); ylabel('molar flow rate (mol/s)');
legend('Cl2', 'P', 'A', 'HCl', 'D');
hold off;

%% Isothermal Equations
function dFdV = Isothermal(~,F, T,P, A,E)
    
    Ftot = sum(F);          % total flow rate [mol/s]
    X = F/Ftot;             % molar fractions
    p = P*X;                % partial pressures [Pa]
    
    kappa = A.*exp(-E/8.314/T);     % kinetic constants [mol/m3/s/Pa]
    r = kappa*p(2)*p(1);            % reaction rates [mol/m3/s]
    R = [-r(1)-r(2), -r(1)-r(2), ...
         r(1), r(1), r(2)]';        % formation rates [mol/m3/s]
    
    dFdV = R;                       % equations [mol/m3/s]
    
end

%% Adiabatic Equations
function dy = Adiabatic(~,y, P,massflow, A,E)
    
    F = y(1:5);             % flow rates [mol/s]
    T = y(6);               % temprature [K]
    Ftot = sum(F);          % total flow rate [mol/s]
    X = F/Ftot;             % molar fractions
    p = P*X;                % partial pressures [Pa]
    
    % Kinetics
    kappa = A.*exp(-E/8.314/T);             % kinetic constants [mol/m3/s/Pa]
    r = kappa*p(2)*p(1);                    % reaction rates [mol/m3/s]
    R = [-r(1)-r(2), -r(1)-r(2), ...
         r(1), r(1), r(2)]';                % formation rates [mol/m3/s]
     
    % Thermodynamics
    [Cp, H] = ThermodynamicProperties(T);   % thermodynamic properties
    CpMol = sum(Cp.*X);                     % mix molar cp (J/mol/K)
    CpMas = CpMol/MW(X);                    % mix mass cp (J/g/K)
    deltaHR = [ H(3)+H(4)-(H(2)+H(1)), ...
                H(5)-(H(2)+H(1))];          % reaction enthalpies [J/mol]

    dFdV = R;                               % equations [mol/m3/s]
    dTdV = -sum(deltaHR.*r)/massflow/CpMas; % equations [K/s]
    
    dy = [dFdV; dTdV];                      % equations
    
end


%% Thermodynamic properties
function [Cp, H] = ThermodynamicProperties(T)

    a = [26.91 3.62 2.52 30.27 10.44];                  % [J/mol/K]
    b = [3.38 23.44 30.44 -0.72 36.52];                 % x10^2 [J/mol/K^2]
    c = [-3.87 -11.59 -22.77 1.24 -26.02];              % x10^5 [J/mol/K^3]
    d = [15.46 22.03 72.88 -3.92 77.36];                % x10^9 [J/mol/K^4]
    Hfref = [0 20096.64 -628.02 -92360.80 -165797.28];  % [J/mol]
    Tref = 298;
    
    Cp = a+(b/1e2)*T+(c/1e5)*T^2+(d/1e9)*T^3;
    H  = Hfref + a*(T-Tref) + (b/1e2/2)*(T^2-Tref^2) + ...
         (c/1e5/3)*(T^3-Tref^3) + (d/1e9/4)*(T^4-Tref^4) ;

end


%% Molecular weight
function MWmix = MW(X)

    % Molecular weights [g/mol]
    MWs = [ 2*35.453, 12*3+6, 35.453+3*12+5, ...
            1+35.453, 3*12+6+2*35.453]; 
    
    MWmix = sum(MWs.*X);
    
end

%% Run isothermal conditions
function [V,F, z,C] = RunIsothermal(P, T, Ftot, npoints)

    global A E ;
    global Ai Vtot;
    global Xin;
    
    Vspan = 0:Vtot/(npoints-1):Vtot;    % integration domain [m3]
    Fin = Xin*Ftot;                     % inlet flow rates [mol/s]
    
    % Solution of ODE system
    options = odeset('RelTol',1e-9,'AbsTol',1e-12);
    [V,F]=ode45(@Isothermal, Vspan, Fin, options, T,P, A,E);
    
    z = V/Ai;               % reactor length [m]
    Ctot = P/8.314/T;       % total concentration [mol/m3]
    Fsum = sum(F,2);        % total flow rate [mol/s]
    C = Ctot*F./Fsum;       % concentrations [mol/m3]
    
end
