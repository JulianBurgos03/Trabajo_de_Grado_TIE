% Limpieza de variables y gráficos previos
close all; clear variables; clc;

% ----%
% FEM %
% ----%
extra = {'ball','solid ball = cylinder(0.2,0.2,0;0.2,0.2,1;0.2) and orthobrick(-1,-1,0;1,1,0.05) -maxh=0.03;'};

% --- Modelo con 8 electrodos ---
fmdl = ng_mk_cyl_models(0, [8], [0.03, 0.05], extra); 
fmdl.stimulation = mk_stim_patterns(8, 1, '{ad}', '{ad}', {'meas_current'}, 0.001);
ctr = interp_mesh(fmdl); 
ctr = (ctr(:,1) - 0.2).^2 + (ctr(:,2) - 0.2).^2;
img = mk_image(fmdl, 1 + 0.1 * (ctr < 0.2^2));
vi1 = fwd_solve(img); % Obtener medidas del modelo de 8 electrodos
%show_fem(img);

% Interpolación para generar 8 Electrodos Virtuales (EV)
vi1_ev = interp1(1:length(vi1.meas), vi1.meas, linspace(1, length(vi1.meas), length(vi1.meas)), 'cubic');

% --- Definir el tamaño de "Conjugada1" ---
tam_8FEM = length(vi1.meas); % Tamaño del vector 8 FEM
tam_8EV = length(vi1_ev);    % Tamaño del vector 8 EV
tam_conjugada1 = tam_8FEM + tam_8EV; % Tamaño total para "Conjugada1"

% --- Construcción de "Conjugada1" con 8 FEM + 8 EV intercalados ---
Conjugada1 = zeros(1, tam_conjugada1)'; % Vector con el tamaño correcto
Conjugada1(1:2:end) = vi1.meas; % Colocar los elementos de 8 FEM en posiciones impares
Conjugada1(2:2:end) = vi1_ev;   % Colocar los elementos de 8 EV en posiciones pares

% --- Interpolación adicional para igualar el tamaño con 16 FEM ---
Conjugada1_interp = interp1(1:tam_conjugada1, Conjugada1, linspace(1, tam_conjugada1, 256), 'cubic')';

% --- Modelo con 16 electrodos ---
fmdl1 = ng_mk_cyl_models(0, [16], [0.03, 0, 0.05], extra);
fmdl1.stimulation = mk_stim_patterns(16, 1, [0,2],'{ad}',{'meas_current'}, 0.001);
ctr = interp_mesh(fmdl1);
ctr = (ctr(:,1) - 0.2).^2 + (ctr(:,2) - 0.2).^2;
img1 = mk_image(fmdl1, 1 + 0.1 * (ctr < 0.2^2));
vi2 = fwd_solve(img1); % Obtener medidas del modelo de 16 electrodos
%figure, show_fem(img1);

% Ajuste a matrix 8x16 de las mediciones reales + inclusiones por método
% lineal
Measurements =reshape(Conjugada1,[16 8])';

% dezplazmiento de las filas de la matriz Measurements
Dezpl_1 = circshift(Measurements(2,:),-1);
Dezpl_2 = circshift(Measurements(3,:),-1);
Dezpl_3 = circshift(Measurements(4,:),-1);
Dezpl_4 = circshift(Measurements(5,:),-1);
Dezpl_5 = circshift(Measurements(6,:),-1);
Dezpl_6 = circshift(Measurements(7,:),-1);
Dezpl_7 = circshift(Measurements(8,:),-1);
Dezpl_8 = circshift(Measurements(8,:),1);
% Interpolación entre las filas de la matriz Measurements para la
% completitud de la matriz de mediciones (16x16)
Conjugada_1= interp1(1:length(Measurements(1,:)), Dezpl_1, linspace(1, length(Measurements(1,:)), 16), 'cubic');
Conjugada_2= interp1(1:length(Measurements(2,:)), Dezpl_2, linspace(1, length(Measurements(2,:)), 16), 'cubic');
Conjugada_3= interp1(1:length(Measurements(3,:)), Dezpl_3, linspace(1, length(Measurements(3,:)), 16), 'cubic');
Conjugada_4= interp1(1:length(Measurements(4,:)), Dezpl_4, linspace(1, length(Measurements(4,:)), 16), 'cubic');
Conjugada_5= interp1(1:length(Measurements(5,:)), Dezpl_5, linspace(1, length(Measurements(5,:)), 16), 'cubic');
Conjugada_6= interp1(1:length(Measurements(6,:)), Dezpl_6, linspace(1, length(Measurements(6,:)), 16), 'cubic');
Conjugada_7= interp1(1:length(Measurements(7,:)), Dezpl_7, linspace(1, length(Measurements(7,:)), 16), 'cubic');
Conjugada_8= interp1(1:length(Measurements(8,:)), Dezpl_8, linspace(1, length(Measurements(8,:)), 16), 'cubic');

offset1 = -0.25E-4;

Conjugada = offset1 +  [Measurements(1,:), Conjugada_1, Measurements(2,:), Conjugada_2, Measurements(3,:), Conjugada_3,...
    Measurements(4,:), Conjugada_4, Measurements(5,:), Conjugada_5, Measurements(6,:), Conjugada_6,...
    Measurements(7,:), Conjugada_7, Measurements(8,:), Conjugada_8]';
% definición de límites máximo y mínimo para eliminar las medicones que
% incluyen los electrodos de inyección.
maximo = 0.002;
minimo = 0;
% Eliminar mediciones que involucran los electrodos de intección
Conjugada(Conjugada > maximo) = [];
Conjugada(Conjugada < minimo ) = [];
vi2.meas(vi2.meas > maximo) = [];
vi2.meas(vi2.meas < minimo ) = [];
% Graficas
figure, plot(vi2.meas), hold on
plot(Conjugada, 'r'), hold off
xlim([0,224])
legend('Mediciones 16 electrodos','Meciones 8 electrodos + 8 inclusiones')

% Error 

Err_Abs = Conjugada-vi2.meas; % error absoluto
Err_Rel = (Conjugada-vi2.meas)./vi2.meas; % error relativo


figure, boxplot(Err_Abs), title ('Error absoluto')
figure, boxplot(Err_Rel), title ('Error relativo')

% Mean absolute error (MAE) (Paper of Jiahao Xu, et al.)

MAE = sum(abs(Err_Abs))./length(Err_Abs)
% Asegurar vectores columna y misma longitud
yhat = Conjugada(:);
yref = vi2.meas(:);
N = min(numel(yhat), numel(yref));
yhat = yhat(1:N); 
yref = yref(1:N);

ER  = norm(yhat - yref, 2) / norm(yref, 2);
MAE = mean(abs(yhat - yref));
CC  = corr(yhat, yref);   % Pearson

fprintf('ER = %.6e\n', ER);
fprintf('MAE = %.6e\n', MAE);
fprintf('CC = %.6f\n', CC);
