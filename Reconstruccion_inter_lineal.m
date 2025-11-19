% SCRIPT DE EVALUACIÓN FINAL DE TESIS (VERSIÓN CON MÉTRICAS Y PLOTEO COMPLETOS)
%
% OBJETIVO:
%   Generar un análisis completo con 3 métricas clave (CC, ER, MAE)
%   para los 3 algoritmos de reconstrucción, comparando los sistemas
%   de 8 electrodos, 8+8 EV, y 16 electrodos. Las figuras se generan
%   para ser visualizadas en pestañas.
%
% AUTOR: Juan José Fernández Pomeo (con asistencia de IA)
% FECHA: Septiembre 2024

%% ========================================================================
%  0. PREPARACIÓN DEL ENTORNO
%% ========================================================================

clear; clc; close all;
try
    eidors_startup;
    fprintf('EIDORS inicializado.\n');
catch ME
    error('Error EIDORS: %s', ME.message);
end
fprintf('=== INICIANDO EVALUACIÓN FINAL CON MÉTRICAS COMPLETAS ===\n\n');

%% ========================================================================
%  1. CONFIGURACIÓN DEL EXPERIMENTO
%% ========================================================================
noise_levels_to_test = [0.05, 0.20];
recon_methods = {'Tikhonov', 'Laplace', 'TV'};
% Eliminamos hp_values fijos y definimos el rango de búsqueda:
hp_range_to_test = logspace(-5, 0, 30); % Rango: [1e-5 a 1] con 30 puntos
phantoms = {
    struct('name', 'Homogeneo', 'inclusions', []), 
    struct('name', 'Central_Pequeno', 'inclusions', {{struct('center', [0, 0], 'radius', 0.2, 'conductivity', 2.0)}}),
    struct('name', 'Central_Grande', 'inclusions', {{struct('center', [0, 0], 'radius', 0.5, 'conductivity', 2.0)}}),
    struct('name', 'Excentrico', 'inclusions', {{struct('center', [0.5, 0.0], 'radius', 0.2, 'conductivity', 2.0)}}),
    struct('name', 'Dual_Simetrico', 'inclusions', {{struct('center', [0.4, 0], 'radius', 0.2, 'conductivity', 2.0), struct('center', [-0.4, 0], 'radius', 0.2, 'conductivity', 2.0)}}),
    struct('name', 'Dual_Asimetrico', 'inclusions', {{struct('center', [0.5, 0.3], 'radius', 0.25, 'conductivity', 2.0), struct('center', [-0.3, -0.4], 'radius', 0.15, 'conductivity', 0.5)}}),
    struct('name', 'Triple', 'inclusions', {{struct('center', [0, 0.5], 'radius', 0.2, 'conductivity', 2.0), struct('center', [0.43, -0.25], 'radius', 0.2, 'conductivity', 2.0), struct('center', [-0.43, -0.25], 'radius', 0.2, 'conductivity', 2.0)}}),
    struct('name', 'Multiples_Esquinas', 'inclusions', {{struct('center', [0.6, 0.6], 'radius', 0.15, 'conductivity', 2.0), struct('center', [-0.6, 0.6], 'radius', 0.15, 'conductivity', 2.0), struct('center', [-0.6, -0.6], 'radius', 0.15, 'conductivity', 2.0), struct('center', [0.6, -0.6], 'radius', 0.15, 'conductivity', 2.0)}}),
    struct('name', 'Anular', 'inclusions', {{struct('shape', 'ring', 'center', [0,0], 'radius_ext', 0.6, 'radius_int', 0.4, 'conductivity', 2.0)}}),
    struct('name', 'Complejo', 'inclusions', {{struct('center', [0,0], 'radius', 0.4, 'conductivity', 1.5), struct('center', [0.5, 0.5], 'radius', 0.15, 'conductivity', 0.2)}})
};
hp_range_to_test = logspace(-4, -1, 20); % Rango de HP a buscar (20 puntos)
results = [];

%% ========================================================================
%  2. BUCLE PRINCIPAL DE EVALUACIÓN
%% ========================================================================
for phantom_def = phantoms'
    current_phantom = phantom_def{:};
    fprintf(['\n', repmat('=', 1, 80), '\n']);
    fprintf('PROCESANDO PHANTOM: %s\n', current_phantom.name);
    
    for nivel_de_ruido = noise_levels_to_test
        fprintf(['--- Ruido: %.1f%% ---\n'], nivel_de_ruido * 100);

        [img_true_16, vh_16, vi_n_16, img_true_8, vh_8, vi_n_8] = simulate_data(current_phantom, nivel_de_ruido);
        [vh_conj, vi_n_conj] = synthesize_16elec_data(vh_8, vi_n_8, vh_16);
        
        plot_data = struct();
        
        for method_name_cell = recon_methods
    method_name = method_name_cell{:};
    fprintf('  > Optimizando HP para %s... ', method_name);

    imdl_8 = mk_common_model('c2c2', 8);
    imdl_16 = mk_common_model('c2c2', 16);

    % Optimizar HP para Línea Base (8)
    hp_8 = find_optimal_hp(imdl_8, vh_8, vi_n_8, method_name, hp_range_to_test);
    
    % Optimizar HP para Conjugado (8+8 EV)
    hp_conj = find_optimal_hp(imdl_16, vh_conj, vi_n_conj, method_name, hp_range_to_test);
    
    % Optimizar HP para Referencia (16)
    hp_16 = find_optimal_hp(imdl_16, vh_16, vi_n_16, method_name, hp_range_to_test);
    
    fprintf('HP Optimos: Base=%.4g, EV=%.4g, Ref=%.4g\n', hp_8, hp_conj, hp_16);
            % --------------------------------------------------------
            fprintf('  > Reconstruyendo con %s... ', method_name);
            
            try
                imdl_rec_8 = configure_imdl(imdl_8, method_name, hp_8);
                img_rec_8 = inv_solve(imdl_rec_8, vh_8, vi_n_8);
                metrics_8 = calculate_metrics_local(img_rec_8, img_true_8);
                
                imdl_rec_conj = configure_imdl(imdl_16, method_name, hp_conj);
                img_rec_conj = inv_solve(imdl_rec_conj, vh_conj, vi_n_conj);
                metrics_conj = calculate_metrics_local(img_rec_conj, img_true_16);
                
                imdl_rec_16 = configure_imdl(imdl_16, method_name, hp_16);
                img_rec_16 = inv_solve(imdl_rec_16, vh_16, vi_n_16);
                metrics_16 = calculate_metrics_local(img_rec_16, img_true_16);
                
                % Guardar TODAS las métricas
                fprintf('  > Reconstruyendo con %s (HP óptimo)... CCs: Base=%.3f, EV=%.3f, Ref=%.3f ✓\n', method_name, metrics_8.CC, metrics_conj.CC, metrics_16.CC);
                results = [results; {current_phantom.name, nivel_de_ruido, method_name, ...
                                     metrics_8.CC, metrics_conj.CC, metrics_16.CC, ...
                                     metrics_8.ER, metrics_conj.ER, metrics_16.ER, ...
                                     metrics_8.MAE, metrics_conj.MAE, metrics_16.MAE}];
                
                fprintf('CCs: Base=%.3f, EV=%.3f, Ref=%.3f ✓\n', metrics_8.CC, metrics_conj.CC, metrics_16.CC);
                
                plot_data.(method_name) = struct('base', img_rec_8, 'conj', img_rec_conj, 'ref', img_rec_16, 'metrics_base', metrics_8, 'metrics_conj', metrics_conj, 'metrics_ref', metrics_16);
            catch ME
                fprintf('FALLÓ: %s ✗\n', ME.message);
                results = [results; {current_phantom.name, nivel_de_ruido, method_name, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN}];
                plot_data.(method_name) = [];
            end
        end
        
        % --- LÓGICA DE PLOTEO EN PESTAÑAS ---
        figure('Name', sprintf('%s - Ruido %.1f%%', current_phantom.name, nivel_de_ruido*100), 'Position', [50 50, 1600, 700]);
    
        all_vals = img_true_16.elem_data(:);
        for i=1:length(recon_methods); method_name=recon_methods{i}; if isfield(plot_data,method_name)&&~isempty(plot_data.(method_name)); all_vals=[all_vals; plot_data.(method_name).base.elem_data(:); plot_data.(method_name).conj.elem_data(:); plot_data.(method_name).ref.elem_data(:)]; end; end
        cmin=min(all_vals); cmax=max(all_vals);

        plot_idx=1;
        for i=1:length(recon_methods)
            method_name = recon_methods{i};
            
            subplot(3, 4, plot_idx); show_fem(img_true_16); title(sprintf('%s\nGround Truth', method_name)); caxis([cmin cmax]); axis equal; axis off; plot_idx=plot_idx+1;
            
            data = plot_data.(method_name);
            if ~isempty(data)
                subplot(3,4,plot_idx); show_fem(data.base); title(sprintf('Línea Base (8)\nCC=%.3f',data.metrics_base.CC)); caxis([cmin cmax]); axis equal; axis off; plot_idx=plot_idx+1;
                subplot(3,4,plot_idx); show_fem(data.conj); title(sprintf('Conjugado (8+8 EV)\nCC=%.3f',data.metrics_conj.CC)); caxis([cmin cmax]); axis equal; axis off; plot_idx=plot_idx+1;
                subplot(3,4,plot_idx); show_fem(data.ref); title(sprintf('Ref. 16 Elec\nCC=%.3f',data.metrics_ref.CC)); caxis([cmin cmax]); axis equal; axis off; plot_idx=plot_idx+1;
            else
                for j=1:3; subplot(3,4,plot_idx); text(0,0,'FALLÓ','Hor','cen','Font',20,'Col','r'); axis off; plot_idx=plot_idx+1; end
            end
        end
        sgtitle(sprintf('Análisis Comparativo | Phantom: %s | Ruido: %.1f%%', strrep(current_phantom.name,'_',' '), nivel_de_ruido*100), 'FontWeight','bold','FontSize',16);
    end
end

%% ========================================================================
%  3. GENERACIÓN DE TABLAS DE RESUMEN COMPLETAS
%% ========================================================================
fprintf(['\n\n', repmat('=', 1, 100), '\n']);
fprintf('GENERANDO TABLAS DE RESUMEN COMPLETAS\n');
fprintf([repmat('=', 1, 100), '\n\n']);

results_table = cell2table(results, 'VariableNames', ...
    {'Phantom', 'Ruido', 'Algoritmo', ...
     'CC_Base', 'CC_EV', 'CC_Ref', ...
     'ER_Base', 'ER_EV', 'ER_Ref', ...
     'MAE_Base', 'MAE_EV', 'MAE_Ref'});

for i = 1:length(recon_methods)
    method_name = recon_methods{i};
    fprintf('--- Tabla de Resumen para: %s ---\n', method_name);
    alg_table = results_table(strcmp(results_table.Algoritmo, method_name), :);
    disp(alg_table);
    fprintf('\n');
end

fprintf('\n=== EVALUACIÓN COMPLETA FINALIZADA ===\n');

%% ========================================================================
%  FUNCIONES LOCALES
%% ========================================================================
function [img_true_16, vh_16, vi_n_16, img_true_8, vh_8, vi_n_8] = simulate_data(current_phantom, nivel_de_ruido)
    imdl_16=mk_common_model('c2c2',16); fmdl_16=imdl_16.fwd_model; img_true_16=create_phantom_local(fmdl_16,current_phantom); vh_16=fwd_solve(mk_image(fmdl_16,1.0)); vi_16=fwd_solve(img_true_16); signal_16=vi_16.meas-vh_16.meas; vi_n_16=vi_16; vi_n_16.meas=vi_16.meas+nivel_de_ruido*std(signal_16)*randn(size(signal_16));
    imdl_8=mk_common_model('c2c2',8); fmdl_8=imdl_8.fwd_model; img_true_8=create_phantom_local(fmdl_8,current_phantom); vh_8=fwd_solve(mk_image(fmdl_8,1.0)); vi_8=fwd_solve(img_true_8); signal_8=vi_8.meas-vh_8.meas; vi_n_8=vi_8; vi_n_8.meas=vi_8.meas+nivel_de_ruido*std(signal_8)*randn(size(signal_8));
end
function [vh_conj, vi_n_conj] = synthesize_16elec_data(vh_8, vi_n_8, vh_16_template)
    % Paso 1: Obtener las diferencias de voltaje (Delta V) en 8 electrodos
    delta_v_8_meas = vi_n_8.meas - vh_8.meas;
    
    n_meas_8=8*(8-3); 
    n_meas_16=16*(16-3); 
    
    % Reorganizar Delta V_8 en matriz de inyecciones x mediciones
    delta_v_8_matrix=reshape(delta_v_8_meas,8-3,8)'; 
    
    % Matriz de Delta V sintetizada para 16 electrodos
    delta_v_16_matrix=zeros(16,16-3); 
    
    for i=1:16
        if mod(i,2)==1; % Electrodos Físicos (Impares)
            idx_8=(i+1)/2; 
            delta_v_fisicas=delta_v_8_matrix(idx_8,:); 
            % Interpolamos Delta V_8 -> Delta V_16 con 'linear'
            delta_v_16_matrix(i,:)=interp1(1:5,delta_v_fisicas,linspace(1,5,13),'linear');
        else; % Electrodos Virtuales (Pares) - Usando promedio de Delta V
            idx_prev=i/2; 
            idx_next=mod(i/2,8)+1; 
            delta_v_virtuales=(delta_v_8_matrix(idx_prev,:)+delta_v_8_matrix(idx_next,:))/2; 
            % Interpolamos Delta V_virtual -> Delta V_16 con 'linear'
            delta_v_16_matrix(i,:)=interp1(1:5,delta_v_virtuales,linspace(1,5,13),'linear');
        end
    end
    
    % Paso 2: Reconstruir V_inclusiones y V_homogeneo para el modelo de 16
    delta_v_conj_meas=reshape(delta_v_16_matrix',n_meas_16,1); 
    
    % Devolver los objetos de voltaje requeridos por inv_solve:
    % vh_conj = V_h (homogéneo) del modelo de 16 (sin interpolación)
    vh_conj = vh_16_template; 
    
    % vi_n_conj = V_i (inclusión) sintetizado
    vi_n_conj = vh_16_template; 
    vi_n_conj.meas = vh_16_template.meas + delta_v_conj_meas; 
    
end
function imdl=configure_imdl(imdl_base,method_name,hp_value)
    imdl=imdl_base; imdl.reconst_type='difference'; imdl.jacobian_bkgnd.value=1.0;
    switch method_name; case 'Tikhonov'; imdl.solve=@inv_solve_diff_GN_one_step; imdl.RtR_prior=@prior_tikhonov; imdl.hyperparameter.value=hp_value; case 'Laplace'; imdl.solve=@inv_solve_diff_GN_one_step; imdl.RtR_prior=@prior_laplace; imdl.hyperparameter.value=hp_value; case 'TV'; imdl.solve=@inv_solve_TV_pdipm; imdl.R_prior=@prior_TV; imdl.hyperparameter.value=hp_value; imdl.parameters.max_iterations=20; end
end
function img=create_phantom_local(fmdl,phantom_def)
    img=mk_image(fmdl,1.0); if isempty(phantom_def.inclusions); return; end; elem_centers=interp_mesh(fmdl); x=elem_centers(:,1); y=elem_centers(:,2);
    for i=1:length(phantom_def.inclusions); inc=phantom_def.inclusions{i}; if isfield(inc,'shape')&&strcmp(inc.shape,'ring'); selector=((x-inc.center(1)).^2+(y-inc.center(2)).^2<inc.radius_ext^2)&((x-inc.center(1)).^2+(y-inc.center(2)).^2>inc.radius_int^2); else; selector=(x-inc.center(1)).^2+(y-inc.center(2)).^2<inc.radius^2; end; img.elem_data(selector)=inc.conductivity; end
end
function metrics=calculate_metrics_local(img_rec,img_true)
    rec_data=img_rec.elem_data; true_data=img_true.elem_data; metrics.ER=norm(rec_data-true_data)/norm(true_data); try; C=corrcoef(rec_data,true_data); metrics.CC=C(1,2); catch; metrics.CC=NaN; end; metrics.RR=sqrt(mean((rec_data-true_data).^2))/sqrt(mean(true_data.^2)); metrics.MAE=mean(abs(rec_data-true_data));
    try; if exist('ssim','file'); grid_res=64; [X,Y]=meshgrid(linspace(-1,1,grid_res)); elem_centers=interp_mesh(img_true.fwd_model); img_true_2d=griddata(elem_centers(:,1),elem_centers(:,2),true_data,X,Y,'linear',{'QJ'}); img_rec_2d=griddata(elem_centers(:,1),elem_centers(:,2),rec_data,X,Y,'linear',{'QJ'}); img_true_2d(isnan(img_true_2d))=1.0; img_rec_2d(isnan(img_rec_2d))=1.0; metrics.SSIM=ssim(img_rec_2d,img_true_2d); else; metrics.SSIM=NaN; end; catch; metrics.SSIM=NaN; end
end

function [optimal_hp] = find_optimal_hp(imdl_base, vh, vi_n, method_name, hp_range)
% FIND_OPTIMAL_HP Encuentra el hiperparámetro óptimo usando la curva L.
%   imdl_base: Modelo inverso base (c2c2 con 8 o 16 electrodos).
%   vh, vi_n: Objetos de voltaje (homogéneo y medido/ruidoso).
%   method_name: 'Tikhonov' o 'Laplace'.
%   hp_range: Vector de hiperparámetros a probar (ej: logspace(-4, -2, 20)).
%
% Nota: Para TV, la curva L es más compleja; por simplicidad, usaremos el mismo 
% rango y encontraremos el punto de compromiso, aunque TV.solve es no-lineal.

    if strcmpi(method_name, 'TV')
        % Advertencia: La curva L no es estrictamente válida para TV no-lineal.
        % Usaremos una aproximación simple.
        fprintf('(Aproximación para TV: Usando el centro del rango)\n');
        optimal_hp = hp_range(ceil(end/2));
        return;
    end
    
    error_norm = zeros(size(hp_range));
    solution_norm = zeros(size(hp_range));
    
    for k = 1:length(hp_range)
        hp = hp_range(k);
        
        % 1. Configurar el modelo inverso (imdl) con el hiperparámetro actual
        imdl = configure_imdl(imdl_base, method_name, hp);
        
        % 2. Resolver la inversión
        try
            img_rec = inv_solve(imdl, vh, vi_n);
            
            % 3. Calcular la norma del error (e.g., ||J*ds - dV||)
            % En EIDORS, 'solve_use_params' facilita esto.
            
            % Nota: Usamos la función de EIDORS que simplifica la curva L
            % ya que calcular ||J*ds - dV|| manualmente es complejo.
            imdl.hyperparameter.value = hp;
            data_solver = feval(imdl.solve, imdl, vh, vi_n);
            
            % Para Tikhonov/Laplace (métodos GN_one_step):
            error_norm(k) = norm(data_solver.meas_diff_norm); % ||J*ds - dV||
            solution_norm(k) = norm(data_solver.reg_norm); % ||R*ds||
            
        catch
            error_norm(k) = NaN;
            solution_norm(k) = NaN;
        end
    end
    
    % Encontrar el punto de la esquina (Corner Point) de la curva L
    % Usando el método de máxima curvatura (cercano al código fuente de EIDORS)
    
    valid_idx = ~isnan(error_norm) & ~isnan(solution_norm);
    x = log(error_norm(valid_idx));
    y = log(solution_norm(valid_idx));
    
    if length(x) < 3
        % No hay suficientes puntos para formar la curva L, usar el centro
        warning('Curva L fallida, usando hiperparámetro central.');
        optimal_hp = hp_range(ceil(end/2));
        return;
    end
    
    % Encontrar la máxima curvatura (método simple/aproximado)
    % Se busca el punto que maximiza la distancia perpendicular a la línea 
    % que conecta los extremos. Esto es una aproximación visual del "codo".
    dx = x(end) - x(1);
    dy = y(end) - y(1);
    
    % Distancia perpendicular de cada punto a la línea extrema
    distances = abs(dy*x - dx*y + x(end)*y(1) - y(end)*x(1)) / sqrt(dx^2 + dy^2);
    
    [~, corner_idx] = max(distances);
    
    % El índice del punto de la esquina en el vector original hp_range
    original_idx = find(valid_idx, 1, 'first') + corner_idx - 1;
    optimal_hp = hp_range(original_idx);
    
end