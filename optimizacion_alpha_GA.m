% optimizacion_alpha_GA.m
% Limpieza
close all; clear; clc;

% ------------------%
% MODELOS FEM       %
% ------------------%
extra = {'ball','solid ball = cylinder(0.2,0.2,0;0.2,0.2,1;0.2) and orthobrick(-1,-1,0;1,1,0.05) -maxh=0.03;'};

% Modelo 8 electrodos
fmdl = ng_mk_cyl_models(0, [8], [0.03, 0.05], extra);
fmdl.stimulation = mk_stim_patterns(8, 1, '{ad}', '{ad}', {'meas_current'}, 0.001);
ctr = interp_mesh(fmdl); ctr = (ctr(:,1)-0.2).^2 + (ctr(:,2)-0.2).^2;
img = mk_image(fmdl, 1 + 0.1*(ctr < 0.2^2));
vi1 = fwd_solve(img);

% Modelo 16 electrodos
fmdl1 = ng_mk_cyl_models(0, [16], [0.03, 0, 0.05], extra);
fmdl1.stimulation = mk_stim_patterns(16, 1, [0,2], '{ad}', {'meas_current'}, 0.001);
ctr = interp_mesh(fmdl1); ctr = (ctr(:,1)-0.2).^2 + (ctr(:,2)-0.2).^2;
img1 = mk_image(fmdl1, 1 + 0.1*(ctr < 0.2^2));
vi2 = fwd_solve(img1);

% Guardar mediciones reales
real_meas = vi2.meas;

% ----------------------------------------%
% FUNCIÓN DE FITNESS PARA EL GA           %
% ----------------------------------------%
fitnessFcn = @(alpha) custom_fitness(alpha, vi1.meas, real_meas);

% --------------------%
% CONFIGURACIÓN GA    %
% --------------------%
options = optimoptions('ga', ...
    'PopulationSize', 50, ...
    'MaxGenerations', 80, ...
    'FunctionTolerance', 1e-6, ...
    'Display', 'iter', ...
    'PlotFcn', {@gaplotbestf});

% Ejecutar GA
[alpha_opt, score_opt] = ga(fitnessFcn, 1, [], [], [], [], 0.01, 0.99, [], options);

% Mostrar resultados
fprintf('\nα óptimo encontrado por GA: %.8f\n', alpha_opt);
fprintf('Score óptimo: %.8f\n', score_opt);

% Regenerar curva óptima con α óptimo
vi1_ev = alpha_opt * (vi1.meas(1:end-1) - vi1.meas(2:end)) + vi1.meas(2:end);
min_len = min(length(vi1.meas), length(vi1_ev));
Conjugada1 = zeros(2*min_len, 1);
Conjugada1(1:2:end) = vi1.meas(1:min_len);
Conjugada1(2:2:end) = vi1_ev(1:min_len);

% Interpolar si es necesario
if length(Conjugada1) < 128
    falta = 128 - length(Conjugada1);
    Conjugada1 = [Conjugada1; interp1(1:length(Conjugada1), Conjugada1, linspace(1, length(Conjugada1), falta), 'linear')'];
end

Measurements = reshape(Conjugada1(1:128), [16, 8])';
Dezpl = @(x, sh) circshift(Measurements(x,:), sh);
C = @(d) interp1(1:16, d, linspace(1, 16, 16), 'linear');
offset1 = -0.25E-4;
Conjugada = offset1 + [ ...
    Measurements(1,:), C(Dezpl(2,-1)), Measurements(2,:), C(Dezpl(3,-1)), ...
    Measurements(3,:), C(Dezpl(4,-1)), Measurements(4,:), C(Dezpl(5,-1)), ...
    Measurements(5,:), C(Dezpl(6,-1)), Measurements(6,:), C(Dezpl(7,-1)), ...
    Measurements(7,:), C(Dezpl(8,-1)), Measurements(8,:), C(Dezpl(8, 1)) ...
]';

% Métricas finales
len = min(length(Conjugada), length(real_meas));
v1 = Conjugada(1:len);
v2 = real_meas(1:len);
ER = norm(v1 - v2) / norm(v2) * 100;
MAE = mean(abs(v1 - v2));
R = corr(v1, v2);

fprintf('\nMétricas con α óptimo:\n');
fprintf('Error Relativo: %.2f%%\n', ER);
fprintf('MAE: %.6f\n', MAE);
fprintf('Correlación: %.4f\n', R);

% Comparación final
figure;
plot(v2, 'b', 'DisplayName', 'Mediciones 16FEM');
hold on;
plot(v1, 'r', 'DisplayName', '8FEM + 8EV (α óptimo)');
legend();
xlabel('Electrodos'); ylabel('Voltaje (V)');
title('Comparación: 16FEM vs. 8FEM + 8EV (\alpha óptimo)');
xlim([0, length(v2)]);
