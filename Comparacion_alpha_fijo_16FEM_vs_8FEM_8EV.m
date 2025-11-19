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
vi1 = fwd_solve(img); % Obtener mediciones del modelo de 8 electrodos

% --- Definir el valor de α para la estimación de electrodos virtuales ---
alpha_12 = 0.049533;  % Definir la posición de los electrodos virtuales (0 < α < 1)

% --- Generar los electrodos virtuales usando la ecuación del artículo ---
num_ev = length(vi1.meas) - 1;  % Se generan n-1 electrodos virtuales
vi1_ev = zeros(num_ev, 1);  % Inicializar el vector con ceros

for i = 1:num_ev
    vi1_ev(i) = alpha_12 * (vi1.meas(i) - vi1.meas(i+1)) + vi1.meas(i+1);
end

% --- Definir el tamaño de "Conjugada1" ---
min_len = min(length(vi1.meas), length(vi1_ev));  % Asegurar que los tamaños coincidan
Conjugada1 = zeros(2 * min_len, 1);  % Vector inicializado en ceros con el tamaño correcto

% Asignar los valores intercalados
Conjugada1(1:2:end) = vi1.meas(1:min_len);  % Electrodos reales en posiciones impares
Conjugada1(2:2:end) = vi1_ev(1:min_len);    % Electrodos virtuales en posiciones pares

% --- Verificar y ajustar la longitud de Conjugada1 ---
fprintf('Tamaño de Conjugada1 después de la intercalación: %d\n', length(Conjugada1));

% Si Conjugada1 tiene menos de 128 elementos, agregamos valores interpolados
if length(Conjugada1) < 128
    falta = 128 - length(Conjugada1);  % Número de valores faltantes
    Conjugada1 = [Conjugada1; interp1(1:length(Conjugada1), Conjugada1, linspace(1, length(Conjugada1), falta), 'linear')'];
end

% --- Ahora Conjugada1 tiene exactamente 128 valores ---
fprintf('Tamaño de Conjugada1 después de ajuste: %d\n', length(Conjugada1));

% --- Ajuste a matriz 8x16 de las mediciones reales + inclusiones por método α ---
Measurements = reshape(Conjugada1(1:128), [16 8])';

% --- Modelo con 16 electrodos ---
fmdl1 = ng_mk_cyl_models(0, [16], [0.03, 0, 0.05], extra);
fmdl1.stimulation = mk_stim_patterns(16, 1, [0,2],'{ad}',{'meas_current'}, 0.001);
ctr = interp_mesh(fmdl1);
ctr = (ctr(:,1) - 0.2).^2 + (ctr(:,2) - 0.2).^2;
img1 = mk_image(fmdl1, 1 + 0.1 * (ctr < 0.2^2));
vi2 = fwd_solve(img1); % Obtener mediciones del modelo de 16 electrodos

% --- Desplazamiento de filas en la matriz Measurements ---
Dezpl_1 = circshift(Measurements(2,:),-1);
Dezpl_2 = circshift(Measurements(3,:),-1);
Dezpl_3 = circshift(Measurements(4,:),-1);
Dezpl_4 = circshift(Measurements(5,:),-1);
Dezpl_5 = circshift(Measurements(6,:),-1);
Dezpl_6 = circshift(Measurements(7,:),-1);
Dezpl_7 = circshift(Measurements(8,:),-1);
Dezpl_8 = circshift(Measurements(8,:),1);

% --- Interpolación entre filas para completar la matriz de mediciones (16x16) ---
Conjugada_1= interp1(1:length(Measurements(1,:)), Dezpl_1, linspace(1, length(Measurements(1,:)), 16), 'linear');
Conjugada_2= interp1(1:length(Measurements(2,:)), Dezpl_2, linspace(1, length(Measurements(2,:)), 16), 'linear');
Conjugada_3= interp1(1:length(Measurements(3,:)), Dezpl_3, linspace(1, length(Measurements(3,:)), 16), 'linear');
Conjugada_4= interp1(1:length(Measurements(4,:)), Dezpl_4, linspace(1, length(Measurements(4,:)), 16), 'linear');
Conjugada_5= interp1(1:length(Measurements(5,:)), Dezpl_5, linspace(1, length(Measurements(5,:)), 16), 'linear');
Conjugada_6= interp1(1:length(Measurements(6,:)), Dezpl_6, linspace(1, length(Measurements(6,:)), 16), 'linear');
Conjugada_7= interp1(1:length(Measurements(7,:)), Dezpl_7, linspace(1, length(Measurements(7,:)), 16), 'linear');
Conjugada_8= interp1(1:length(Measurements(8,:)), Dezpl_8, linspace(1, length(Measurements(8,:)), 16), 'linear');

offset1 = -0.25E-4;

Conjugada = offset1 +  [Measurements(1,:), Conjugada_1, Measurements(2,:), Conjugada_2, Measurements(3,:), Conjugada_3,...
    Measurements(4,:), Conjugada_4, Measurements(5,:), Conjugada_5, Measurements(6,:), Conjugada_6,...
    Measurements(7,:), Conjugada_7, Measurements(8,:), Conjugada_8]';

% --- Eliminación de valores fuera del rango permitido ---
maximo = 0.002;
minimo = 0;
Conjugada(Conjugada > maximo) = [];
Conjugada(Conjugada < minimo ) = [];
vi2.meas(vi2.meas > maximo) = [];
vi2.meas(vi2.meas < minimo ) = [];

% --- Graficar Comparación de Mediciones 16FEM vs. 8FEM+8EV ---
figure;
plot(vi2.meas, 'b', 'DisplayName', 'Mediciones 16 electrodos');
hold on;
plot(Conjugada, 'r', 'DisplayName', 'Mediciones 8 electrodos + 8 inclusiones');
hold off;
xlim([0,224])
legend();
title('Comparación de Mediciones 16FEM vs. 8FEM+8EV');
xlabel('Electrodos');
ylabel('Voltaje (V)');
% --- Métricas: ER, MAE y CC ---
y_hat = Conjugada(:);      % estimado (8FEM + 8EV)
y_ref = vi2.meas(:);       % referencia (16FEM)

% Alinear longitudes por si el filtrado eliminó índices distintos
L = min(numel(y_hat), numel(y_ref));
y_hat = y_hat(1:L);
y_ref = y_ref(1:L);

diffv = y_hat - y_ref;

% ER relativo global (norma-2), MAE y coeficiente de correlación (Pearson)
ER  = norm(diffv, 2) / norm(y_ref, 2);
MAE = mean(abs(diffv));
R   = corrcoef(y_hat, y_ref);
CC  = R(1,2);

fprintf('ER  = %.6e\n', ER);
fprintf('MAE = %.6e\n', MAE);
fprintf('CC  = %.6f\n',  CC);

% (Opcional) Boxplots de errores
figure; boxplot(diffv); title('Error absoluto'); ylabel('Voltaje (V)');
figure; boxplot(diffv ./ y_ref); title('Error relativo punto a punto'); ylabel('Relativo');
