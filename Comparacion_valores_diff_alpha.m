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

% --- Definir los valores de α para la estimación de electrodos virtuales ---
alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7];  % Diferentes valores de α

% --- Inicialización de almacenamiento ---
vi1_ev_all = cell(length(alpha_values), 1);

% --- Generar los electrodos virtuales usando la ecuación del artículo ---
num_ev = length(vi1.meas) - 1;  % Se generan n-1 electrodos virtuales

for idx = 1:length(alpha_values)
    alpha_12 = alpha_values(idx);
    vi1_ev = zeros(num_ev, 1);  % Inicializar el vector con ceros
    
    for i = 1:num_ev
        vi1_ev(i) = alpha_12 * (vi1.meas(i) - vi1.meas(i+1)) + vi1.meas(i+1);
    end
    
    vi1_ev_all{idx} = vi1_ev; % Guardar resultados
end

% --- Gráfica para visualizar la estimación de electrodos virtuales ---
figure;
hold on;
plot(1:length(vi1.meas), vi1.meas, 'ko-', 'LineWidth', 1.5, 'DisplayName', 'Electrodos Reales');

colors = {'r', 'g', 'b', 'm', 'c'};
markers = {'o', 's', '^', 'd', 'x'};

for idx = 1:length(alpha_values)
    plot(1.5:1:length(vi1_ev_all{idx})+0.5, vi1_ev_all{idx}, ...
        [colors{idx} markers{idx} '-'], 'LineWidth', 1, ...
        'DisplayName', sprintf('\\alpha = %.1f', alpha_values(idx)));
end

hold off;
legend();
title('Comparación de Electrodos Virtuales con Diferentes \alpha_{1-2}');
xlabel('Posición del Electrodo');
ylabel('Voltaje (V)');

% --- Cálculo de errores ---
ER_values = [];
correlation_values = [];
MAE_values = [];

for idx = 1:length(alpha_values)
    vi1_ev_current = vi1_ev_all{idx};
    
    % Calcular Error Relativo (ER) en porcentaje
    ER = mean(abs((vi1_ev_current - vi1.meas(1:length(vi1_ev_current))) ./ vi1.meas(1:length(vi1_ev_current)))) * 100;
    
    % Calcular Coeficiente de Correlación de Pearson
    correlation = corr(vi1_ev_current, vi1.meas(1:length(vi1_ev_current)));
    
    % Calcular Error Medio Absoluto (MAE)
    MAE = mean(abs(vi1_ev_current - vi1.meas(1:length(vi1_ev_current))));
    
    % Guardar valores
    ER_values = [ER_values, ER];
    correlation_values = [correlation_values, correlation];
    MAE_values = [MAE_values, MAE];
end

% --- Mostrar resultados en consola ---
fprintf('\nResultados de Error para cada α:\n');
for idx = 1:length(alpha_values)
    fprintf('Alpha = %.1f --> ER: %.2f%% | Correlación: %.4f | MAE: %.6f\n', ...
        alpha_values(idx), ER_values(idx), correlation_values(idx), MAE_values(idx));
end

