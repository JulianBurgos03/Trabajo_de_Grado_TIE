%% ═══════════════════════════════════════════════════════════════════
%% PIPELINE: GENERACIÓN FÍSICA DE ELECTRODOS VIRTUALES + RECONSTRUCCIONES
%% ═══════════════════════════════════════════════════════════════════
close all; clear; clc; rng(42);

fprintf('╔══════════════════════════════════════════════════════╗\n');
fprintf('║  GENERACIÓN FÍSICA DE ELECTRODOS VIRTUALES (v2.0)   ║\n');
fprintf('╚══════════════════════════════════════════════════════╝\n\n');

%% PARTE 1-6: TU CÓDIGO ORIGINAL (SIN CAMBIOS)
%% ════════════════════════════════════════════════════════════════════

data_path = 'C:\Users\A15\Downloads\DatosMar\';

datasets = {
    'Referencia_5grSal_D2.mat',      'Referencia',  'ref';
    'CuerpoConductor_5grSal_D2.mat', 'Conductor',   'train';
    'CuerpoResistivo_5grSal_D2.mat', 'Resistivo',   'train';
    'Zanahoria_5grSal_D2.mat',       'Zanahoria',   'val'
};

data_store = struct();

fprintf('PARTE 1: CARGA DE DATOS\n');
for d = 1:size(datasets, 1)
    fname = datasets{d, 1};
    name = datasets{d, 2};
    split = datasets{d, 3};
    
    fprintf('  [%d/%d] Cargando %s (%s)... ', d, size(datasets,1), name, split);
    
    obj_data = load([data_path, fname]);
    n_frames = length(obj_data.current_data);
    
    measurements_8 = zeros(n_frames, 8);
    for frame = 1:n_frames
        try
            mux0 = obj_data.current_data(frame).data(1);
            measurements_8(frame, :) = mux0.data(1, :);
        catch
        end
    end
    
    for e = 1:8
        col = measurements_8(:, e);
        mu = mean(col);
        sigma = std(col);
        outliers = (col > mu + 5*sigma) | (col < mu - 5*sigma);
        if any(outliers)
            measurements_8(outliers, e) = mu;
        end
    end
    
    data_store.(lower(name)) = struct(...
        'name', name, ...
        'split', split, ...
        'data_8fem', measurements_8, ...
        'n_frames', n_frames);
    
    fprintf('OK (%d frames)\n', n_frames);
end

fprintf('\n');

% PARTE 2-6: Interpolación, evaluación, validación...
% (Todo tu código de las partes 2-6 aquí - sin cambios)

function V_virt = physical_interpolation(V_8fem, method)
    n_frames = size(V_8fem, 1);
    V_virt = zeros(n_frames, 8);
    
    for v = 1:8
        left = v;
        right = mod(v, 8) + 1;
        
        V_left = V_8fem(:, left);
        V_right = V_8fem(:, right);
        
        switch method
            case 'linear'
                V_virt(:, v) = 0.5 * (V_left + V_right);
            case 'weighted'
                grad = abs(V_right - V_left);
                max_grad = max(grad);
                if max_grad > 0
                    weights = 1 - (grad / max_grad);
                    alpha_left = 0.5 + 0.2 * weights;
                    alpha_right = 1 - alpha_left;
                else
                    alpha_left = 0.5;
                    alpha_right = 0.5;
                end
                V_virt(:, v) = alpha_left .* V_left + alpha_right .* V_right;
            case 'gradient'
                left2 = mod(v - 2, 8) + 1;
                if left2 < 1, left2 = left2 + 8; end
                right2 = mod(v + 1, 8) + 1;
                if right2 > 8, right2 = right2 - 8; end
                
                V_left2 = V_8fem(:, left2);
                V_right2 = V_8fem(:, right2);
                V_base = 0.5 * (V_left + V_right);
                curvature = 0.25 * (V_left2 - 2*V_left + V_right + V_left + V_right - 2*V_right2);
                V_virt(:, v) = V_base + 0.1 * curvature;
        end
    end
end

methods = {'linear', 'weighted', 'gradient'};
results_baseline = struct();

fprintf('PARTE 2: INTERPOLACIÓN\n');
for m = 1:length(methods)
    method_name = methods{m};
    fprintf('  Método: %s\n', upper(method_name));
    
    for d = 1:size(datasets, 1)
        name = lower(datasets{d, 2});
        if strcmp(name, 'referencia'), continue; end
        
        V_8fem = data_store.(name).data_8fem;
        V_8virt = physical_interpolation(V_8fem, method_name);
        
        n_frames = size(V_8fem, 1);
        V_16total = zeros(n_frames, 16);
        for k = 1:8
            V_16total(:, 2*k-1) = V_8fem(:, k);
            V_16total(:, 2*k) = V_8virt(:, k);
        end
        
        if ~isfield(results_baseline, method_name)
            results_baseline.(method_name) = struct();
        end
        results_baseline.(method_name).(name) = V_16total;
    end
end

fprintf('\nPARTE 3: EVALUACIÓN\n');
eval_table = zeros(length(methods), 4);

for m = 1:length(methods)
    method_name = methods{m};
    
    corr_spatial_all = [];
    smoothness_all = [];
    consistency_all = [];
    
    for d = 1:size(datasets, 1)
        name = lower(datasets{d, 2});
        if strcmp(name, 'referencia'), continue; end
        
        V_8fem = data_store.(name).data_8fem;
        V_16total = results_baseline.(method_name).(name);
        
        corr_vec = zeros(16, 1);
        for e = 1:16
            e_next = mod(e, 16) + 1;
            corr_vec(e) = corr(V_16total(:, e), V_16total(:, e_next));
        end
        corr_spatial_all = [corr_spatial_all; mean(corr_vec)];
        
        smoothness_vec = zeros(8, 1);
        for v = 1:8
            virtual_col = V_16total(:, 2*v);
            diffs = diff(virtual_col);
            smoothness_vec(v) = std(diffs);
        end
        smoothness_all = [smoothness_all; mean(smoothness_vec)];
        
        consistency_vec = zeros(8, 1);
        for v = 1:8
            left_phys = V_8fem(:, v);
            right_phys = V_8fem(:, mod(v, 8) + 1);
            virtual = V_16total(:, 2*v);
            within_bounds = (virtual >= min(left_phys, right_phys)) & (virtual <= max(left_phys, right_phys));
            consistency_vec(v) = sum(within_bounds) / length(virtual);
        end
        consistency_all = [consistency_all; mean(consistency_vec)];
    end
    
    mean_corr = mean(corr_spatial_all);
    mean_smooth = mean(smoothness_all);
    mean_consist = mean(consistency_all);
    global_score = 0.4*mean_corr + 0.3*(1 - mean_smooth/max(smoothness_all)) + 0.3*mean_consist;
    
    eval_table(m, :) = [mean_corr, mean_smooth, mean_consist, global_score];
    
    fprintf('  %s: Score=%.4f\n', method_name, global_score);
end

[~, best_idx] = max(eval_table(:, 4));
best_method = methods{best_idx};
fprintf('  Mejor: %s\n\n', upper(best_method));

fprintf('PARTE 4: GENERAR VIRTUALES FINALES\n');
final_results = struct();

for d = 1:size(datasets, 1)
    name = lower(datasets{d, 2});
    V_8fem = data_store.(name).data_8fem;
    V_8virt = physical_interpolation(V_8fem, best_method);
    
    n_frames = size(V_8fem, 1);
    V_16total = zeros(n_frames, 16);
    for k = 1:8
        V_16total(:, 2*k-1) = V_8fem(:, k);
        V_16total(:, 2*k) = V_8virt(:, k);
    end
    
    final_results.(name) = struct('V_8fem', V_8fem, 'V_8virt', V_8virt, 'V_16total', V_16total);
    fprintf('  %s OK\n', datasets{d, 2});
end

fprintf('\nPARTE 5: VALIDACIÓN\n');
V_val_8 = final_results.zanahoria.V_8fem;
V_val_virt = final_results.zanahoria.V_8virt;

er_vals = zeros(8, 1);
cc_vals = zeros(8, 1);

for v = 1:8
    left = v;
    right = mod(v, 8) + 1;
    V_gt = 0.5 * (V_val_8(:, left) + V_val_8(:, right));
    V_pred = V_val_virt(:, v);
    er_vals(v) = norm(V_pred - V_gt) / (norm(V_gt) + eps);
    cc_vals(v) = corr(V_pred, V_gt);
end

fprintf('  ER: %.4f ± %.4f\n', mean(er_vals), std(er_vals));
fprintf('  CC: %.4f ± %.4f\n\n', mean(cc_vals), std(cc_vals));

%% ═══════════════════════════════════════════════════════════════════
%% PARTE 7 CORREGIDA: RECONSTRUCCIONES EIT
%% ═══════════════════════════════════════════════════════════════════
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('PARTE 7: RECONSTRUCCIONES EIT (VERSIÓN CORREGIDA)\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

%% 7.1) Inicializar EIDORS
fprintf('7.1) Inicializando EIDORS...\n');
try
    run('C:\Users\A15\Documents\MATLAB\eidors-v3.11-ng\eidors\startup.m');
    fprintf('   ✓ EIDORS inicializado\n\n');
catch ME
    error('No se pudo inicializar EIDORS: %s', ME.message);
end

%% 7.2) CREAR MODELOS FEM (GEOMETRÍA NORMALIZADA)
fprintf('7.2) Creando modelos FEM...\n');

% ✅ USAR GEOMETRÍA NORMALIZADA (estándar EIDORS)
tank_radius = 1.0;
tank_height = 1.0;  % Aumentado para estabilidad
electrode_height = 0.1;

fprintf('   Geometría normalizada: R=%.1f, H=%.1f\n', tank_radius, tank_height);

% Modelo 8 electrodos
fprintf('   [1/2] Modelo 8 electrodos... ');
fmdl_8 = ng_mk_cyl_models([tank_height, tank_radius, electrode_height], [8, 1], [0.05]);
fmdl_8.stimulation = mk_stim_patterns(8, 1, [0, 1], [0, 1], {}, 1);
n_meas_8 = size(fmdl_8.stimulation(1).meas_pattern, 1);
fprintf('OK (%d meas/patrón)\n', n_meas_8);

% Modelo 16 electrodos
fprintf('   [2/2] Modelo 16 electrodos... ');
fmdl_16 = ng_mk_cyl_models([tank_height, tank_radius, electrode_height], [16, 1], [0.05]);
fmdl_16.stimulation = mk_stim_patterns(16, 1, [0, 1], [0, 1], {}, 1);
n_meas_16 = size(fmdl_16.stimulation(1).meas_pattern, 1);
fprintf('OK (%d meas/patrón)\n\n', n_meas_16);

%% 7.3) HIPERPARÁMETROS
lambda_8 = 0.05;
lambda_16 = 0.01;

imdl_8 = select_imdl(fmdl_8, {'Basic GN dif'});
imdl_8.hyperparameter.value = lambda_8;
imdl_8.RtR_prior = @prior_noser;
imdl_8.solve = @inv_solve_diff_GN_one_step;

imdl_16 = select_imdl(fmdl_16, {'Basic GN dif'});
imdl_16.hyperparameter.value = lambda_16;
imdl_16.RtR_prior = @prior_noser;
imdl_16.solve = @inv_solve_diff_GN_one_step;

fprintf('   ✓ λ_8=%.3f, λ_16=%.3f\n\n', lambda_8, lambda_16);

%% 7.4) PREPARAR MEDICIONES DESDE DATOS REALES
fprintf('7.4) Convirtiendo voltajes experimentales...\n');

test_measurements = struct();

for d = 1:size(datasets, 1)
    name = lower(datasets{d, 2});
    
    V_8_all = final_results.(name).V_8fem;
    V_16_all = final_results.(name).V_16total;
    
    % Seleccionar frame con mejor SNR
    if strcmp(name, 'referencia')
        % Para referencia: promedio de todos los frames
        V_8_frame = mean(V_8_all, 1);
        V_16_frame = mean(V_16_all, 1);
        frame_idx = 0;
    else
        % Para objetos: frame con máximo contraste
        snr_frames = std(V_8_all, 0, 2) ./ (mean(abs(V_8_all), 2) + eps);
        [~, frame_idx] = max(snr_frames);
        V_8_frame = V_8_all(frame_idx, :);
        V_16_frame = V_16_all(frame_idx, :);
    end
    
    % ✅ CONVERSIÓN CLAVE: Voltajes → Mediciones Diferenciales
    meas_8 = experimental_to_eidors_measurements(V_8_frame, 8);
    meas_16 = experimental_to_eidors_measurements(V_16_frame, 16);
    
    test_measurements.(name) = struct(...
        'meas_8', meas_8, ...
        'meas_16', meas_16, ...
        'frame', frame_idx);
    
    fprintf('   %s: %d meas (8e), %d meas (16e)\n', ...
        datasets{d, 2}, length(meas_8), length(meas_16));
end

fprintf('\n');

%% 7.5) RECONSTRUIR DESDE DATOS REALES
fprintf('7.5) Reconstruyendo desde mediciones experimentales...\n');

% ✅ BASELINE = Referencia (tanque homogéneo)
vh_8 = test_measurements.referencia.meas_8;
vh_16 = test_measurements.referencia.meas_16;

recons = struct();
objects = {'conductor', 'resistivo', 'zanahoria'};

for o = 1:length(objects)
    name = objects{o};
    fprintf('   [%d/%d] %s... ', o, length(objects), name);
    
    try
        % Mediciones del objeto
        vi_8 = test_measurements.(name).meas_8;
        vi_16 = test_measurements.(name).meas_16;
        
        % Reconstruir
        img_8 = inv_solve(imdl_8, vh_8, vi_8);
        img_16 = inv_solve(imdl_16, vh_16, vi_16);
        
        recons.(name) = struct('img_8', img_8, 'img_16', img_16, 'name', name);
        
        fprintf('OK\n');
        fprintf('       8 FEM:   σ ∈ [%.3f, %.3f], mean=%.3f, std=%.3f\n', ...
            min(img_8.elem_data), max(img_8.elem_data), ...
            mean(img_8.elem_data), std(img_8.elem_data));
        fprintf('       16 Total: σ ∈ [%.3f, %.3f], mean=%.3f, std=%.3f\n', ...
            min(img_16.elem_data), max(img_16.elem_data), ...
            mean(img_16.elem_data), std(img_16.elem_data));
    catch ME
        fprintf('ERROR: %s\n', ME.message);
        recons.(name) = struct('img_8', [], 'img_16', [], 'name', name);
    end
end

fprintf('\n');

%% 7.6) Análisis
fprintf('7.6) Análisis cuantitativo...\n');

results_eit = zeros(length(objects), 5);

for o = 1:length(objects)
    name = objects{o};
    if isempty(recons.(name).img_8), continue; end
    
    sigma_8 = recons.(name).img_8.elem_data;
    sigma_16 = recons.(name).img_16.elem_data;
    
    % Métricas
    std_8 = std(sigma_8);
    std_16 = std(sigma_16);
    contrast_improv = 100 * (std_16 - std_8) / std_8;
    
    snr_8 = abs(mean(sigma_8)) / (std_8 + eps);
    snr_16 = abs(mean(sigma_16)) / (std_16 + eps);
    snr_improv = 100 * (snr_16 - snr_8) / (snr_8 + eps);
    
    try
        cc = compute_spatial_corr(recons.(name).img_8, recons.(name).img_16);
    catch
        cc = corr(sigma_8, sigma_16);
    end
    
    results_eit(o, :) = [contrast_improv, snr_improv, cc, (contrast_improv + snr_improv)/2, 1];
    
    fprintf('   %s: ΔContraste=%+.1f%%, ΔSNR=%+.1f%%, CC=%.3f\n', ...
        name, contrast_improv, snr_improv, cc);
end

fprintf('\n');

%% 7.7) Visualizar
fprintf('7.7) Visualizaciones...\n');

eit_dir = 'resultados_eit_corregidos';
if ~exist(eit_dir, 'dir'), mkdir(eit_dir); end

for o = 1:length(objects)
    name = objects{o};
    if isempty(recons.(name).img_8), continue; end
    
    fig = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');
    
    subplot(1, 3, 1);
    show_fem(recons.(name).img_8);
    title(sprintf('%s - 8 FEM', upper(name)), 'FontSize', 14, 'FontWeight', 'bold');
    axis equal tight; colorbar;
    
    subplot(1, 3, 2);
    show_fem(recons.(name).img_16);
    title(sprintf('%s - 16 Total', upper(name)), 'FontSize', 14, 'FontWeight', 'bold');
    axis equal tight; colorbar;
    
    subplot(1, 3, 3);
    hold on;
    histogram(recons.(name).img_8.elem_data, 30, 'FaceColor', [0.3, 0.6, 0.9], 'FaceAlpha', 0.5);
    histogram(recons.(name).img_16.elem_data, 30, 'FaceColor', [0.9, 0.5, 0.3], 'FaceAlpha', 0.5);
    hold off;
    legend('8 FEM', '16 Total');
    xlabel('σ'); ylabel('Prob'); title('Distribución');
    grid on;
    
    saveas(fig, sprintf('%s/%s.png', eit_dir, name));
    close(fig);
    
    fprintf('   %s OK\n', name);
end

fprintf('\n✓ Guardado en: %s/\n\n', eit_dir);

%% 7.8) Guardar
save('resultados_corregidos.mat', 'recons', 'results_eit', 'final_results', ...
    'lambda_8', 'lambda_16', '-v7.3');

fprintf('╔══════════════════════════════════════════════════════╗\n');
fprintf('║ 🎉 COMPLETADO                                        ║\n');
fprintf('╠══════════════════════════════════════════════════════╣\n');
fprintf('║ λ_8  = %.3f (corregido: MÁS regularización)       ║\n', lambda_8);
fprintf('║ λ_16 = %.3f (corregido: MENOS regularización)     ║\n', lambda_16);
fprintf('╚══════════════════════════════════════════════════════╝\n\n');

%% FUNCIONES AUXILIARES

function img = simulate_phantom(fmdl, phantom_type)
    img = mk_image(fmdl, 1.0);
    nodes = fmdl.nodes;
    elems = fmdl.elems;
    
    centers = zeros(size(elems, 1), 3);
    for i = 1:size(elems, 1)
        centers(i, :) = mean(nodes(elems(i, 1:3), :), 1);
    end
    
    radii = sqrt(centers(:,1).^2 + centers(:,2).^2);
    
    switch lower(phantom_type)
        case 'conductor'
            center_elems = find(radii <= 0.2);
            img.elem_data(center_elems) = 2.0;
        case 'resistivo'
            center_elems = find(radii <= 0.2);
            img.elem_data(center_elems) = 0.5;
        case 'zanahoria'
            center_elems = find(radii <= 0.25);
            img.elem_data(center_elems) = 1.5 + 0.3*randn(length(center_elems), 1);
            img.elem_data(img.elem_data < 0.5) = 0.5;
    end
end

function cc = compute_spatial_corr(img_8, img_16)
    try
        nodes_8 = img_8.fwd_model.nodes;
        nodes_16 = img_16.fwd_model.nodes;
        elems_8 = img_8.fwd_model.elems;
        elems_16 = img_16.fwd_model.elems;
        
        centers_8 = zeros(size(elems_8, 1), 2);
        for i = 1:size(elems_8, 1)
            centers_8(i, :) = mean(nodes_8(elems_8(i, 1:3), 1:2), 1);
        end
        
        centers_16 = zeros(size(elems_16, 1), 2);
        for i = 1:size(elems_16, 1)
            centers_16(i, :) = mean(nodes_16(elems_16(i, 1:3), 1:2), 1);
        end
        
        [X, Y] = meshgrid(linspace(-0.9, 0.9, 64), linspace(-0.9, 0.9, 64));
        
        sigma_8_grid = griddata(centers_8(:,1), centers_8(:,2), img_8.elem_data, X, Y, 'natural');
        sigma_16_grid = griddata(centers_16(:,1), centers_16(:,2), img_16.elem_data, X, Y, 'natural');
        
        R = sqrt(X.^2 + Y.^2);
        mask = (R <= 1.0) & ~isnan(sigma_8_grid) & ~isnan(sigma_16_grid);
        
        if sum(mask(:)) > 30
            cc = corr(sigma_8_grid(mask), sigma_16_grid(mask));
        else
            cc = NaN;
        end
    catch
        cc = corr(img_8.elem_data, img_16.elem_data);
    end
end
function meas = voltages_to_measurements(voltages, fmdl)
    % Convierte voltajes a mediciones diferenciales según patrón de fmdl
    
    n_elec = length(voltages);
    stim = fmdl.stimulation;
    n_stim = length(stim);
    
    meas = [];
    
    for s = 1:n_stim
        meas_pat = stim(s).meas_pattern;
        n_meas = size(meas_pat, 1);
        
        for m = 1:n_meas
            % Electrodos positivo y negativo
            elec_pos = find(meas_pat(m, :) > 0);
            elec_neg = find(meas_pat(m, :) < 0);
            
            if ~isempty(elec_pos) && ~isempty(elec_neg)
                v_diff = voltages(elec_pos(1)) - voltages(elec_neg(1));
                meas = [meas; v_diff];
            end
        end
    end
end
function meas = experimental_to_eidors_measurements(voltages, n_elec)
    % Convierte voltajes experimentales a mediciones diferenciales
    % compatibles con EIDORS (patrón adjacent-drive, adjacent-skip)
    
    n_patterns = n_elec;
    n_meas_per_pattern = n_elec - 3;  % Skip adjacent electrodes
    
    meas = zeros(n_patterns * n_meas_per_pattern, 1);
    idx = 1;
    
    for stim = 1:n_patterns
        % Electrodos de inyección
        inject_pos = stim;
        inject_neg = mod(stim, n_elec) + 1;
        
        % Medir en todos menos adyacentes
        for m = 1:n_meas_per_pattern
            % Electrodo positivo de medición (skip 1 después de inject_neg)
            meas_pos = mod(inject_neg + m, n_elec) + 1;
            if meas_pos < 1, meas_pos = meas_pos + n_elec; end
            if meas_pos > n_elec, meas_pos = meas_pos - n_elec; end
            
            % Electrodo negativo de medición
            meas_neg = mod(meas_pos, n_elec) + 1;
            
            % Diferencia de voltaje
            v_diff = voltages(meas_pos) - voltages(meas_neg);
            
            meas(idx) = v_diff;
            idx = idx + 1;
        end
    end
end