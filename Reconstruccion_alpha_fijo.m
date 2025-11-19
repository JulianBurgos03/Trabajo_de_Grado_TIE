% SCRIPT DE EVALUACIÓN FINAL DE TESIS - VERSIÓN CON OPTIMIZACIÓN DE ALPHA
%
% ANÁLISIS COMPARATIVO:
%   1. Comparación de métodos de optimización de α (Fixed, GridSearch, GA manual)
%   2. Comparación de algoritmos de reconstrucción (Tikhonov, Laplace, TV)
%   3. Métricas: CC, ER, MAE
%   4. Análisis de resultados con tablas detalladas
%
% AUTOR: Juan José Fernández Pomeo
% FECHA: Octubre 2025

%% ========================================================================
%  0. PREPARACIÓN DEL ENTORNO
%% ========================================================================

clear; clc; close all;
try
    eidors_startup;
    fprintf('✓ EIDORS inicializado correctamente.\n');
catch ME
    error('✗ Error EIDORS: %s', ME.message);
end

fprintf('\n=== INICIANDO EVALUACIÓN FINAL CON OPTIMIZACIÓN DE α ===\n\n');

%% ========================================================================
%  1. CONFIGURACIÓN DEL EXPERIMENTO
%% ========================================================================

noise_levels_to_test = [0.05, 0.20];
recon_methods = {'Tikhonov', 'Laplace', 'TV'};
hp_values = struct('Tikhonov', 3e-3, 'Laplace', 6e-3, 'TV', 3e-4);
alpha_methods = {'Fixed', 'GridSearch', 'GA'};

phantoms = {
    struct('name', 'Homogeneo', 'inclusions', []), 
    struct('name', 'Central_Pequeno', 'inclusions', {{struct('center', [0, 0], 'radius', 0.2, 'conductivity', 2.0)}}),
    struct('name', 'Central_Grande', 'inclusions', {{struct('center', [0, 0], 'radius', 0.5, 'conductivity', 2.0)}}),
    struct('name', 'Excentrico', 'inclusions', {{struct('center', [0.5, 0.0], 'radius', 0.2, 'conductivity', 2.0)}}),
    struct('name', 'Dual_Simetrico', 'inclusions', {{struct('center', [0.4, 0], 'radius', 0.2, 'conductivity', 2.0), struct('center', [-0.4, 0], 'radius', 0.2, 'conductivity', 2.0)}}),
    struct('name', 'Triple', 'inclusions', {{struct('center', [0, 0.5], 'radius', 0.2, 'conductivity', 2.0), struct('center', [0.43, -0.25], 'radius', 0.2, 'conductivity', 2.0), struct('center', [-0.43, -0.25], 'radius', 0.2, 'conductivity', 2.0)}})
};

results = [];
alpha_optimization_results = [];

%% ========================================================================
%  2. BUCLE PRINCIPAL DE EVALUACIÓN
%% ========================================================================

for phantom_def = phantoms'
    current_phantom = phantom_def{:};
    fprintf(['\n', repmat('=', 1, 100), '\n']);
    fprintf('PROCESANDO PHANTOM: %s\n', current_phantom.name);
    fprintf([repmat('=', 1, 100), '\n']);
    
    for nivel_de_ruido = noise_levels_to_test
        fprintf('\n--- Nivel de Ruido: %.1f%% ---\n', nivel_de_ruido * 100);

        [img_true_16, vh_16, vi_n_16, img_true_8, vh_8, vi_n_8] = simulate_data(current_phantom, nivel_de_ruido);
        fprintf('  Dimensiones: 8elec=%d meas | 16elec=%d meas\n', length(vh_8.meas), length(vh_16.meas));
        
        for method_name_cell = recon_methods
            method_name = method_name_cell{:};
            hp_value = hp_values.(method_name);
            
            fprintf('\n  Procesando algoritmo: %s\n', method_name);
            
            % --- LÍNEA BASE Y REFERENCIA (se calculan una vez por algoritmo) ---
            try
                imdl_8 = mk_common_model('c2c2', 8);
                imdl_rec_8_base = configure_imdl(imdl_8, method_name, hp_value);
                img_rec_8 = inv_solve(imdl_rec_8_base, vh_8, vi_n_8);
                metrics_8 = calculate_metrics_local(img_rec_8, img_true_8);
                fprintf('    Línea Base (8 Elec): CC=%.3f | ER=%.3f | MAE=%.4f\n', metrics_8.CC, metrics_8.ER, metrics_8.MAE);
            catch ME; fprintf('    ✗ Falló línea base: %s\n', ME.message); metrics_8 = struct('CC', NaN, 'ER', NaN, 'MAE', NaN); end
            
            try
                imdl_16_base = mk_common_model('c2c2', 16);
                imdl_rec_16_ref = configure_imdl(imdl_16_base, method_name, hp_value);
                img_rec_16 = inv_solve(imdl_rec_16_ref, vh_16, vi_n_16);
                metrics_16 = calculate_metrics_local(img_rec_16, img_true_16);
                fprintf('    Referencia (16 Elec):CC=%.3f | ER=%.3f | MAE=%.4f\n', metrics_16.CC, metrics_16.ER, metrics_16.MAE);
            catch ME; fprintf('    ✗ Falló referencia: %s\n', ME.message); metrics_16 = struct('CC', NaN, 'ER', NaN, 'MAE', NaN); end
            
            % --- OPTIMIZACIÓN DE α Y RECONSTRUCCIÓN CON EVs ---
            for alpha_method_cell = alpha_methods
                alpha_method = alpha_method_cell{:};
                
                tic;
                imdl_rec_16_for_opt = configure_imdl(mk_common_model('c2c2', 16), method_name, hp_value);
                
                switch alpha_method
                    case 'Fixed'
                        alpha_opt = 0.5;
                    case 'GridSearch'
                        alpha_opt = optimize_alpha_grid_search(vh_8, vi_n_8, vh_16, img_true_16, imdl_rec_16_for_opt);
                    case 'GA'
                        alpha_opt = optimize_alpha_ga_manual(vh_8, vi_n_8, vh_16, img_true_16, imdl_rec_16_for_opt);
                end
                opt_time = toc;
                
                [vh_conj, vi_n_conj] = synthesize_16elec_data(vh_8, vi_n_8, vh_16, alpha_opt);
                
                try
                    img_rec_conj = inv_solve(imdl_rec_16_for_opt, vh_conj, vi_n_conj);
                    metrics_conj = calculate_metrics_local(img_rec_conj, img_true_16);

                    % >>> INICIO ALMACENAMIENTO DE IMÁGENES CLAVE <<<
                    phantoms_to_save = {'Central_Grande', 'Excentrico', 'Triple'};
                    
                    if ismember(current_phantom.name, phantoms_to_save) && ...
                       strcmp(alpha_method, 'GridSearch') && ...
                       nivel_de_ruido == 0.05
                        
                        phantom_key = current_phantom.name;
                        method_key = method_name; % Guardar para todos los algoritmos
                        
                        viz_results.(phantom_key).(method_key).img_true = img_true_16;
                        viz_results.(phantom_key).(method_key).img_ref = img_rec_16;
                        viz_results.(phantom_key).(method_key).img_ev = img_rec_conj;
                    end
                    % >>> FIN ALMACENAMIENTO DE IMÁGENES CLAVE <<<

                    fprintf('      → %-10s: α=%.3f | CC=%.3f | ER=%.3f | MAE=%.4f (%.2fs)\n', ...
                        alpha_method, alpha_opt, metrics_conj.CC, metrics_conj.ER, metrics_conj.MAE, opt_time);
                catch ME
                    fprintf('    ✗ Falló %s: %s\n', alpha_method, ME.message);
                    metrics_conj = struct('CC', NaN, 'ER', NaN, 'MAE', NaN);
                end
                
                results = [results; {current_phantom.name, nivel_de_ruido, method_name, alpha_method, ...
                                     metrics_8.CC, metrics_conj.CC, metrics_16.CC, ...
                                     metrics_8.ER, metrics_conj.ER, metrics_16.ER, ...
                                     metrics_8.MAE, metrics_conj.MAE, metrics_16.MAE}];
                
                alpha_optimization_results = [alpha_optimization_results; {current_phantom.name, nivel_de_ruido, method_name, alpha_method, alpha_opt, opt_time, metrics_conj.CC}];
            end
        end
    end
end

%% ========================================================================
%  3. ANÁLISIS DE RESULTADOS
%% ========================================================================
% (El código de la sección 3 se mantiene igual que en tu versión,
% ya que es excelente para el análisis).
fprintf(['\n\n', repmat('=', 1, 100), '\n']);
fprintf('ANÁLISIS DE RESULTADOS\n');
fprintf([repmat('=', 1, 100), '\n\n']);
results_table = cell2table(results, 'VariableNames', {'Phantom', 'Ruido', 'Algoritmo', 'Metodo_Alpha', 'CC_Base', 'CC_EV', 'CC_Ref', 'ER_Base', 'ER_EV', 'ER_Ref', 'MAE_Base', 'MAE_EV', 'MAE_Ref'});
alpha_table = cell2table(alpha_optimization_results, 'VariableNames', {'Phantom', 'Ruido', 'Algoritmo', 'Metodo_Alpha', 'Alpha_Optimo', 'Tiempo_s', 'CC_Obtenido'});
fprintf('═══ TABLA 1: COMPARACIÓN POR ALGORITMO DE RECONSTRUCCIÓN ═══\n\n');
for i=1:length(recon_methods); method_name=recon_methods{i}; fprintf('--- %s ---\n',method_name); subset=results_table(strcmp(results_table.Algoritmo,method_name),:); for alpha_method_cell=alpha_methods; alpha_method=alpha_method_cell{:}; alpha_subset=subset(strcmp(subset.Metodo_Alpha,alpha_method),:); if ~isempty(alpha_subset); avg_cc_ev=nanmean(alpha_subset.CC_EV); avg_cc_ref=nanmean(alpha_subset.CC_Ref); avg_er_ev=nanmean(alpha_subset.ER_EV); time_subset=alpha_table(strcmp(alpha_table.Algoritmo,method_name)&strcmp(alpha_table.Metodo_Alpha,alpha_method),:); avg_time=nanmean(time_subset.Tiempo_s); fprintf('  %15s: CC_EV=%.4f (vs Ref=%.4f) | ER=%.4f | Tiempo=%.2fs\n',alpha_method,avg_cc_ev,avg_cc_ref,avg_er_ev,avg_time); end; end; fprintf('\n'); end
fprintf('\n═══ TABLA 2: VALORES DE α ÓPTIMO POR MÉTODO ═══\n\n');
for alpha_method_cell=alpha_methods; alpha_method=alpha_method_cell{:}; if strcmp(alpha_method,'Fixed'); continue; end; fprintf('--- %s ---\n',alpha_method); alpha_subset=alpha_table(strcmp(alpha_table.Metodo_Alpha,alpha_method),:); fprintf('  Media α: %.4f ± %.4f\n',nanmean(alpha_subset.Alpha_Optimo),nanstd(alpha_subset.Alpha_Optimo)); fprintf('  Rango α: [%.4f, %.4f]\n',min(alpha_subset.Alpha_Optimo),max(alpha_subset.Alpha_Optimo)); fprintf('  Tiempo promedio: %.2f s\n\n',nanmean(alpha_subset.Tiempo_s)); end
fprintf('═══ TABLA 3: ANÁLISIS ESTADÍSTICO (8+8 EV vs 16 Real) ═══\n\n');
for i=1:length(recon_methods); method_name=recon_methods{i}; for alpha_method_cell=alpha_methods; alpha_method=alpha_method_cell{:}; subset=results_table(strcmp(results_table.Algoritmo,method_name)&strcmp(results_table.Metodo_Alpha,alpha_method),:); delta_CC=subset.CC_EV-subset.CC_Ref; delta_ER=subset.ER_EV-subset.ER_Ref; try; [p_cc,h_cc]=signrank(delta_CC); [p_er,h_er]=signrank(delta_ER); fprintf('Método: %s - %s\n',method_name,alpha_method); fprintf('  Δ CC: %.4f ± %.4f | p-val: %.4e %s\n',nanmean(delta_CC),nanstd(delta_CC),p_cc,iff(h_cc,'*','')); fprintf('  Δ ER: %.4f ± %.4f | p-val: %.4e %s\n\n',nanmean(delta_ER),nanstd(delta_ER),p_er,iff(h_er,'*','')); catch; fprintf('Método: %s - %s - Error en test\n\n',method_name,alpha_method); end; end; end
fprintf('* = significativo (p < 0.05)\n'); fprintf('\n=== EVALUACIÓN COMPLETA FINALIZADA ===\n');

%  4. VISUALIZACIÓN DE RESULTADOS CLAVE MULTI-PHANTOM (Loop de Figuras)
%% ========================================================================
% Establecemos los phantoms que se van a visualizar y el método a usar
% (Usamos Laplace, ya que mostró el mejor rendimiento con EVs)
if exist('viz_results', 'var') && isfield(viz_results, 'Central_Grande')
    
    phantoms_to_plot = {'Central_Grande', 'Excentrico', 'Triple'};
    methods_to_use = {'Laplace'}; 
    
    fprintf(['\n\n', repmat('═', 1, 100), '\n']);
    fprintf('4. GENERANDO VISUALIZACIONES SECUENCIALES (GridSearch, Laplace, 5.0%% Ruido)\n');
    fprintf([repmat('═', 1, 100), '\n']);
    
    for phantom_name_cell = phantoms_to_plot
        phantom_name = phantom_name_cell{:};
        m_name = methods_to_use{1}; % Usar Laplace
        
        % Comprobar que los datos existen para este phantom y método
        if ~isfield(viz_results, phantom_name) || ~isfield(viz_results.(phantom_name), m_name);
            continue; 
        end
        
        data = viz_results.(phantom_name).(m_name);
        
        % Recuperar métricas del caso (GridSearch, 0.05 ruido)
        case_results = results_table(strcmp(results_table.Phantom, phantom_name) & ...
                                     strcmp(results_table.Algoritmo, m_name) & ...
                                     strcmp(results_table.Metodo_Alpha, 'GridSearch') & ...
                                     results_table.Ruido == 0.05, :);
        
        cc_ev_val = case_results.CC_EV(1);
        cc_ref_val = case_results.CC_Ref(1);

        % Crear una nueva figura para este phantom
        figure('Position', [100 100 1200 400], 'Name', sprintf('Reconstrucción: %s', phantom_name));
        
        % Columna 1: Ground Truth
        subplot(1, 3, 1);
        show_fem(data.img_true);
        title('1. Ground Truth', 'FontWeight', 'bold', 'FontSize', 12);
        ylabel(strrep(phantom_name, '_', ' '), 'FontWeight', 'bold', 'FontSize', 14);
        axis equal; axis tight;
        
        % Columna 2: 8E + 8EV (Optimizado)
        subplot(1, 3, 2);
        show_fem(data.img_ev);
        title(sprintf('2. EV Opt. (CC=%.4f)', cc_ev_val), 'FontWeight', 'bold', 'FontSize', 12);
        axis equal; axis tight;

        % Columna 3: 16E Referencia
        subplot(1, 3, 3);
        show_fem(data.img_ref);
        title(sprintf('3. 16E Ref (CC=%.4f)', cc_ref_val), 'FontWeight', 'bold', 'FontSize', 12);
        axis equal; axis tight;
        
        colormap(jet);
        sgtitle(sprintf('Reconstrucción %s con %s (EV Opt. vs 16E Ref.)', strrep(phantom_name, '_', ' '), m_name), 'FontSize', 16);
        fprintf('  ✓ Figura generada para Phantom: %s\n', phantom_name);
    end

else
    fprintf('\n\n4. VISUALIZACIÓN: No se pudo generar la imagen comparativa (viz_results no definida o vacía).\n');
end

%% ========================================================================
%  FUNCIONES AUXILIARES
% ========================================================================
function result=iff(condition,true_val,false_val); if nargin<3;false_val='';end;if condition;result=true_val;else;result=false_val;end;end
function [img_true_16, vh_16, vi_n_16, img_true_8, vh_8, vi_n_8] = simulate_data(current_phantom, nivel_de_ruido)
    imdl_16 = mk_common_model('c2c2', 16); fmdl_16 = imdl_16.fwd_model; img_true_16 = create_phantom_local(fmdl_16, current_phantom); vh_16 = fwd_solve(mk_image(fmdl_16, 1.0)); vi_16 = fwd_solve(img_true_16); signal_16 = vi_16.meas - vh_16.meas; vi_n_16 = vi_16; vi_n_16.meas = vi_16.meas + nivel_de_ruido * std(signal_16) * randn(size(signal_16));
    imdl_8 = mk_common_model('c2c2', 8); fmdl_8 = imdl_8.fwd_model; img_true_8 = create_phantom_local(fmdl_8, current_phantom); vh_8 = fwd_solve(mk_image(fmdl_8, 1.0)); vi_8 = fwd_solve(img_true_8); signal_8 = vi_8.meas - vh_8.meas; vi_n_8 = vi_8; vi_n_8.meas = vi_8.meas + nivel_de_ruido * std(signal_8) * randn(size(signal_8));
end
function [vh_conj, vi_n_conj] = synthesize_16elec_data(vh_8, vi_n_8, vh_16_template, alpha)
    if nargin < 4; alpha = 0.5; end
    
    N_meas_8E = length(vh_8.meas);
    N_meas_16E_target = length(vh_16_template.meas); % 256

    % 1. Cerrar el círculo para interpolación
    vh_meas_circ = [vh_8.meas; vh_8.meas(1)]; 
    vi_n_meas_circ = [vi_n_8.meas; vi_n_8.meas(1)];

    vh_ev = zeros(N_meas_8E, 1);
    vi_n_ev = zeros(N_meas_8E, 1);

    % 2. Generar Electrodos Virtuales (EV) con la fórmula Laplace-Alpha
    for i = 1:N_meas_8E
        % Fórmula V_i' = alpha * (V_i - V_{i+1}) + V_{i+1}
        vh_ev(i) = alpha * (vh_meas_circ(i) - vh_meas_circ(i+1)) + vh_meas_circ(i+1);
        vi_n_ev(i) = alpha * (vi_n_meas_circ(i) - vi_n_meas_circ(i+1)) + vi_n_meas_circ(i+1);
    end

    % 3. Intercalación (Físico, Virtual, Físico, Virtual...)
    % Resulta en un vector de tamaño 2 * N_meas_8E (ej: 128)
    N_intercalado = 2 * N_meas_8E;
    
    V_conj_intercalado = zeros(N_intercalado, 1); 
    V_conj_intercalado(1:2:end) = vh_8.meas; % V_i
    V_conj_intercalado(2:2:end) = vh_ev;     % V'_i

    Vi_n_conj_intercalado = zeros(N_intercalado, 1); 
    Vi_n_conj_intercalado(1:2:end) = vi_n_8.meas; % V_i
    Vi_n_conj_intercalado(2:2:end) = vi_n_ev;     % V'_i

    % 4. Interpolación Final a 256 (Si es necesario)
    % El patrón 'ad' para 8 electrodos produce 64 mediciones. 2*64 = 128. 
    % El patrón [0,2] para 16 electrodos produce 256 mediciones.
    
    % Se requiere la lógica de 'Desplazamiento e Interpolación' de la Sección 3
    % (que es compleja) o una interpolación lineal simple forzada a 256.
    % Usaremos interpolación lineal para este esquema simplificado:
    
    vh_conj_meas = interp1(1:N_intercalado, V_conj_intercalado, ...
                           linspace(1, N_intercalado, N_meas_16E_target), 'linear')';
    vi_n_conj_meas = interp1(1:N_intercalado, Vi_n_conj_intercalado, ...
                             linspace(1, N_intercalado, N_meas_16E_target), 'linear')';
    
    % 5. Asignación a estructuras EIDORS (mantener el FMDL de 16E)
    vh_conj = vh_16_template;
    vh_conj.meas = vh_conj_meas;
    
    vi_n_conj = vh_16_template;
    vi_n_conj.meas = vi_n_conj_meas;
end
function alpha_opt = optimize_alpha_grid_search(vh_8, vi_n_8, vh_16_template, img_true_16, imdl_rec_16)
    alpha_range = linspace(0.1, 0.9, 9); best_alpha = 0.5; best_CC = -Inf;
    for alpha_test = alpha_range; try; [vh_test, vi_test] = synthesize_16elec_data(vh_8, vi_n_8, vh_16_template, alpha_test); img_test = inv_solve(imdl_rec_16, vh_test, vi_test); metrics_test = calculate_metrics_local(img_test, img_true_16); if ~isnan(metrics_test.CC) && metrics_test.CC > best_CC; best_CC = metrics_test.CC; best_alpha = alpha_test; end; catch; continue; end; end
    alpha_opt = best_alpha;
end
function alpha_opt = optimize_alpha_ga_manual(vh_8, vi_n_8, vh_16_template, img_true_16, imdl_rec_16)
    pop_size=10; n_generations=5; mutation_rate=0.1; elite_size=2;
    population=0.1+0.8*rand(pop_size,1);
    for gen=1:n_generations
        fitness=zeros(pop_size,1); for i=1:pop_size; fitness(i)=evaluate_alpha_fitness(population(i),vh_8,vi_n_8,vh_16_template,img_true_16,imdl_rec_16); end
        [~,idx_sorted]=sort(fitness,'descend'); population=population(idx_sorted);
        new_population=zeros(pop_size,1); new_population(1:elite_size)=population(1:elite_size);
        for i=(elite_size+1):pop_size
            p1_idx=randi(pop_size); p2_idx=randi(pop_size); parent1=population(p1_idx); parent2=population(p2_idx);
            child=0.5*(parent1+parent2); if rand()<mutation_rate; child=child+0.1*randn(); child=max(0.1,min(0.9,child)); end
            new_population(i)=child;
        end
        population=new_population;
    end
    final_fitness=zeros(pop_size,1); for i=1:pop_size; final_fitness(i)=evaluate_alpha_fitness(population(i),vh_8,vi_n_8,vh_16_template,img_true_16,imdl_rec_16); end
    [~,best_idx]=max(final_fitness); alpha_opt=population(best_idx);
end
function fitness = evaluate_alpha_fitness(alpha, vh_8, vi_n_8, vh_16_template, img_true_16, imdl_rec_16)
    try; [vh_test,vi_test]=synthesize_16elec_data(vh_8,vi_n_8,vh_16_template,alpha); img_test=inv_solve(imdl_rec_16,vh_test,vi_test); metrics_test=calculate_metrics_local(img_test,img_true_16); fitness=metrics_test.CC; catch; fitness=-Inf; end
    if isnan(fitness); fitness=-Inf; end
end
function imdl=configure_imdl(imdl_base,method_name,hp_value)
    imdl=imdl_base; imdl.reconst_type='difference'; imdl.jacobian_bkgnd.value=1.0;
    switch method_name; case 'Tikhonov';imdl.solve=@inv_solve_diff_GN_one_step; imdl.RtR_prior=@prior_tikhonov; imdl.hyperparameter.value=hp_value; case 'Laplace'; imdl.solve=@inv_solve_diff_GN_one_step; imdl.RtR_prior=@prior_laplace; imdl.hyperparameter.value=hp_value; case 'TV'; imdl.solve=@inv_solve_TV_pdipm; imdl.R_prior=@prior_TV; imdl.hyperparameter.value=hp_value; imdl.parameters.max_iterations=20; end
end
function img=create_phantom_local(fmdl,phantom_def)
    img=mk_image(fmdl,1.0); if isempty(phantom_def.inclusions);return;end; elem_centers=interp_mesh(fmdl); x=elem_centers(:,1);y=elem_centers(:,2);
    for i=1:length(phantom_def.inclusions); inc=phantom_def.inclusions{i}; if isfield(inc,'shape')&&strcmp(inc.shape,'ring'); selector=((x-inc.center(1)).^2+(y-inc.center(2)).^2<inc.radius_ext^2)&((x-inc.center(1)).^2+(y-inc.center(2)).^2>inc.radius_int^2); else; selector=(x-inc.center(1)).^2+(y-inc.center(2)).^2<inc.radius^2; end; img.elem_data(selector)=inc.conductivity; end
end
function metrics=calculate_metrics_local(img_rec,img_true)
    if isempty(img_rec)||isempty(img_true); metrics=struct('CC',NaN,'ER',NaN,'MAE',NaN); return; end
    rec_data=img_rec.elem_data; true_data=img_true.elem_data; metrics.ER=norm(rec_data-true_data)/norm(true_data);
    try; C=corrcoef(rec_data,true_data); metrics.CC=C(1,2); catch; metrics.CC=NaN; end; metrics.MAE=mean(abs(rec_data-true_data));
end