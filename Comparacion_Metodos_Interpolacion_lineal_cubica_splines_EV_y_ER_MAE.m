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
vi1 = fwd_solve(img); % Obtener medidas del 000 modelo de 8 electrodos

% --- Definir el valor de α para la estimación de electrodos virtuales ---
alpha_12 = 0.5;  % Definir la posición de los electrodos virtuales (0 < α < 1)

% --- Generar los electrodos virtuales usando la ecuación del artículo ---
num_ev = length(vi1.meas) - 1;  % Se generan n-1 electrodos virtuales
vi1_ev = zeros(num_ev, 1);  % Inicializar el vector con ceros

for i = 1:num_ev
    vi1_ev(i) = alpha_12 * (vi1.meas(i) - vi1.meas(i+1)) + vi1.meas(i+1);
end

% --- Definir el tamaño de "Conjugada1" ---
tam_8FEM = length(vi1.meas);  % Tamaño de 8 FEM
tam_8EV = length(vi1_ev);  % Tamaño de 8 EV
tam_conjugada1 = tam_8FEM + tam_8EV;  % Total FEM + EV

% --- Construcción de "Conjugada1" con 8 FEM + 8 EV intercalados ---
Conjugada1 = zeros(tam_conjugada1, 1);  % Vector inicializado en ceros
Conjugada1(1:2:end) = vi1.meas;  % Electrodos reales en posiciones impares
Conjugada1(2:2:end) = vi1_ev;    % Electrodos virtuales en posiciones pares

% --- Interpolación con diferentes métodos ---
x_original = 1:length(Conjugada1);
x_interpolado = linspace(1, length(Conjugada1), 256);

vi_lineal = interp1(x_original, Conjugada1, x_interpolado, 'linear');
vi_cubico = interp1(x_original, Conjugada1, x_interpolado, 'cubic');
vi_spline = interp1(x_original, Conjugada1, x_interpolado, 'spline');

% --- Modelo con 16 electrodos ---
fmdl1 = ng_mk_cyl_models(0, [16], [0.03, 0, 0.05], extra);
fmdl1.stimulation = mk_stim_patterns(16, 1, [0,2],'{ad}',{'meas_current'}, 0.001);
ctr = interp_mesh(fmdl1);
ctr = (ctr(:,1) - 0.2).^2 + (ctr(:,2) - 0.2).^2;
img1 = mk_image(fmdl1, 1 + 0.1 * (ctr < 0.2^2));
vi2 = fwd_solve(img1); % Obtener mediciones del modelo de 16 electrodos
vi_real = vi2.meas; % Mediciones reales con 16 electrodos

% --- Cálculo de errores ---

% Error Relativo (ER)
ER_lineal = norm(vi_real - vi_lineal') / norm(vi_real) * 100;
ER_cubico = norm(vi_real - vi_cubico') / norm(vi_real) * 100;
ER_spline = norm(vi_real - vi_spline') / norm(vi_real) * 100;

% Coeficiente de correlación (Pearson)
corr_lineal = corrcoef(vi_real, vi_lineal');
corr_cubico = corrcoef(vi_real, vi_cubico');
corr_spline = corrcoef(vi_real, vi_spline');

% Error Medio Absoluto (MAE)
MAE_lineal = mean(abs(vi_real - vi_lineal'));
MAE_cubico = mean(abs(vi_real - vi_cubico'));
MAE_spline = mean(abs(vi_real - vi_spline'));

% --- Mostrar resultados ---
fprintf('Error Relativo (ER) en porcentaje:\n');
fprintf('Lineal: %.2f%% | Cúbico: %.2f%% | Spline: %.2f%%\n', ER_lineal, ER_cubico, ER_spline);

fprintf('\nCoeficiente de Correlación (Pearson):\n');
fprintf('Lineal: %.4f | Cúbico: %.4f | Spline: %.4f\n', corr_lineal(1,2), corr_cubico(1,2), corr_spline(1,2));

fprintf('\nError Medio Absoluto (MAE):\n');
fprintf('Lineal: %.6f | Cúbico: %.6f | Spline: %.6f\n', MAE_lineal, MAE_cubico, MAE_spline);

% --- Gráfica para comparar los métodos de interpolación ---
figure;
plot(x_original, Conjugada1, 'ko-', 'DisplayName', 'Electrodos Reales');
hold on;
plot(x_interpolado, vi_lineal, 'b*-', 'DisplayName', 'Interpolación Lineal');
plot(x_interpolado, vi_spline, 'm+-', 'DisplayName', 'Interpolación Spline');
plot(x_interpolado, vi_cubico, 'g^-', 'DisplayName', 'Interpolación Cúbica');
hold off;
legend();
title('Comparación de Métodos de Interpolación para Electrodos Virtuales');
xlabel('Posición del Electrodo');
ylabel('Voltaje (V)');
