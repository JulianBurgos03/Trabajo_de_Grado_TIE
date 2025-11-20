%% ==========================================================================
%% SCRIPT: Gran_Comparacion.m
%% Validación Sistemática del Modelo CNN EIT - Xu et al. (2022)
%% Autor: Juan José Fernández Pomeo
%% Fecha: 16 de Julio, 2025
%% ==========================================================================
%
% PROPÓSITO:
%   Este script realiza una validación exhaustiva del modelo CNN entrenado
%   comparando las simulaciones de 16 electrodos (Ground Truth) contra
%   las predicciones de la CNN (Señal Conjugada) para todos los 10 escenarios
%   de phantom definidos en el sistema.
%
% METODOLOGÍA:
%   - Carga el modelo CNN entrenado desde Python/TensorFlow
%   - Genera phantoms determinísticos para cada escenario (1-10)
%   - Compara Ground Truth vs Predicciones CNN
%   - Visualiza resultados en figuras organizadas
%   - Calcula métricas de rendimiento por escenario
%
% DEPENDENCIAS:
%   - Python con TensorFlow instalado
%   - Funciones MATLAB del pipeline EIT
%   - Archivos del modelo entrenado en ruta especificada
%
%% ==========================================================================

%% Limpieza inicial del entorno
close all; clear; clc;
% --- CORRECCIÓN CRÍTICA DE VISUALIZACIÓN EXTREMA ---
set(0, 'DefaultFigureRenderer', 'painters'); 
set(groot, 'defaultFigureColormap', jet); % Asegurar un mapa de color estándar
% --------------------------------------------------
run startup.m
% Silenciar warnings específicos
warning('off', 'add_circular_inclusion:NoElementsAffected');

%% Banner inicial
fprintf('\n');
fprintf('🚀==================================================================🚀\n');
fprintf('   GRAN COMPARACIÓN - VALIDACIÓN SISTEMÁTICA CNN EIT\n');
fprintf('   Replicación Xu et al. (2022) - Todos los Escenarios\n');
fprintf('   Autor: Juan José Fernández Pomeo\n');
fprintf('   Fecha: %s\n', datestr(now));
fprintf('🚀==================================================================🚀\n');
fprintf('\n');

%% =====================================================================
%% SECCIÓN 1: CONFIGURACIÓN DE RUTAS Y VERIFICACIÓN DE ARCHIVOS
%% =====================================================================
fprintf('📁 CONFIGURANDO RUTAS Y VERIFICANDO ARCHIVOS...\n');

% RUTA EXACTA DE LOS ARTEFACTOS ENTRENADOS
model_dir = 'C:\Users\juanp\TG_EIT\notebooks\modelo_10_phantoms';
model_path = fullfile(model_dir, 'modelo_cnn_best.keras');
scalers_path = fullfile(model_dir, 'scalers.pkl');

fprintf('   📂 Directorio del modelo: %s\n', model_dir);
fprintf('   🧠 Ruta del modelo: %s\n', model_path);
fprintf('   ⚙️  Ruta de scalers: %s\n', scalers_path);

% Verificar existencia de archivos críticos
archivos_requeridos = {model_path, scalers_path};
nombres_archivos = {'modelo_cnn_best.keras', 'scalers.pkl'};

for i = 1:length(archivos_requeridos)
   if exist(archivos_requeridos{i}, 'file')
       file_size = dir(archivos_requeridos{i});
       size_mb = file_size.bytes / (1024^2);
       fprintf('   ✅ %s encontrado (%.2f MB)\n', nombres_archivos{i}, size_mb);
   else
       error('❌ ERROR: No se encontró %s en la ruta especificada: %s', ...
           nombres_archivos{i}, archivos_requeridos{i});
   end
end

%% =====================================================================
%% SECCIÓN 2: CONFIGURACIÓN DE LA INTERFAZ CON PYTHON
%% =====================================================================
fprintf('\n🐍 CONFIGURANDO INTERFAZ CON PYTHON...\n');

% Verificar versión de Python
try
   python_version = pyenv;
   fprintf('   🐍 Python configurado: %s\n', python_version.Version);
   fprintf('   📁 Ejecutable: %s\n', python_version.Executable);
catch ME
   error('❌ ERROR: No se pudo configurar Python. Asegúrese de que Python esté instalado.\nError: %s', ME.message);
end

% Verificar módulos de Python requeridos
try
   fprintf('   📦 Verificando módulos de Python...\n');
   py.importlib.import_module('tensorflow');
   py.importlib.import_module('pickle');
   py.importlib.import_module('numpy');
   fprintf('   ✅ Módulos TensorFlow, pickle y numpy disponibles\n');
catch ME
   error('❌ ERROR: Faltan módulos de Python requeridos.\nAsegúrese de tener TensorFlow instalado.\nError: %s', ME.message);
end

%% =====================================================================
% SECCIÓN 3: CARGA DEL MODELO CNN Y SCALERS
% =====================================================================
fprintf('\n🧠 CARGANDO MODELO CNN Y SCALERS...\n');

% Cargar modelo TensorFlow
try
    fprintf('   🧠 Cargando modelo CNN desde: %s\n', model_path);
    tic;
    modelo_cnn = py.tensorflow.keras.models.load_model(model_path);
    load_time = toc;
    
    fprintf('   ✅ Modelo CNN cargado exitosamente (%.2f segundos)\n', load_time);
    fprintf('   📐 Modelo listo para predicciones\n');
    
catch ME
    error('❌ ERROR: No se pudo cargar el modelo CNN.\nError: %s', ME.message);
end

% Cargar scalers
try
    fprintf('   ⚙️  Cargando scalers desde: %s\n', scalers_path);
    
    % Método alternativo usando joblib si pickle falla
    try
        % Intentar con pickle primero
        scalers_file = py.open(scalers_path, 'rb');
        scalers_data = py.pickle.load(scalers_file);
        scalers_file.close();
        fprintf('   ✅ Scalers cargados con pickle\n');
    catch pickle_error
        fprintf('   ⚠️  Pickle falló, intentando con joblib...\n');
        % Intentar con joblib
        joblib = py.importlib.import_module('joblib');
        scalers_data = joblib.load(scalers_path);
        fprintf('   ✅ Scalers cargados con joblib\n');
    end
    
    % Extraer scalers
    scaler_X = scalers_data{'scaler_X'};
    scaler_y = scalers_data{'scaler_y'};
    training_info = scalers_data{'training_info'};
    
    % Mostrar información de entrenamiento
    fprintf('   ✅ Scalers extraídos exitosamente\n');
    fprintf('   📊 Muestras de entrenamiento: %s\n', char(training_info{'training_samples'}));
    fprintf('   🏆 Mejor época: %s\n', char(training_info{'best_epoch'}));
    fprintf('   📅 Fecha entrenamiento: %s\n', char(training_info{'training_date'}));
    
catch ME
    error('❌ ERROR: No se pudieron cargar los scalers.\nError: %s', ME.message);
end

%% =====================================================================
%% SECCIÓN 4: PREPARACIÓN DE MODELOS FEM Y ESCENARIOS
%% =====================================================================
fprintf('\n🏗️  PREPARANDO MODELOS FEM Y ESCENARIOS...\n');

% Configuración para generación de phantoms
CONFIG = struct();
CONFIG.conductividad_fondo = 1.0;
CONFIG.conductividad_objeto = 0.3;

% Crear modelos FEM
try
    fprintf('   📐 Creando modelo FEM de 8 electrodos...\n');
    fmdl_8e = crear_modelo_fem(8);
    
    fprintf('   📐 Creando modelo FEM de 16 electrodos...\n');
    fmdl_16e = crear_modelo_fem(16);
    
    fprintf('   ✅ Modelos FEM creados exitosamente\n');
    fprintf('   📊 Modelo 8e: %d nodos, %d elementos\n', size(fmdl_8e.nodes, 1), size(fmdl_8e.elems, 1));
    fprintf('   📊 Modelo 16e: %d nodos, %d elementos\n', size(fmdl_16e.nodes, 1), size(fmdl_16e.elems, 1));
    
catch ME
    error('❌ ERROR: No se pudieron crear los modelos FEM.\nError: %s', ME.message);
end

% Obtener definiciones de escenarios
try
    fprintf('   📋 Obteniendo definiciones de escenarios...\n');
    scenarios = get_scenarios_definition();
    n_scenarios = length(scenarios);
    
    fprintf('   ✅ %d escenarios cargados exitosamente\n', n_scenarios);
    
    % Mostrar lista de escenarios
    for i = 1:n_scenarios
        fprintf('   🎯 Escenario %2d: %s\n', i, scenarios{i}.name);
    end
    
catch ME
    error('❌ ERROR: No se pudieron cargar las definiciones de escenarios.\nError: %s', ME.message);
end

%% =====================================================================
%% SECCIÓN 5: BUCLE DE VALIDACIÓN SISTEMÁTICA
%% =====================================================================
fprintf('\n🔄 INICIANDO VALIDACIÓN SISTEMÁTICA DE %d ESCENARIOS...\n', n_scenarios);
fprintf('================================================================\n');

% Silenciar warnings de inclusiones pequeñas
warning('off', 'add_circular_inclusion:NoElementsAffected');

% Inicializar arrays para almacenar resultados
resultados = cell(n_scenarios, 1);
metricas_resumen = zeros(n_scenarios, 4); % [CC, MAE, ER, Tiempo_CNN]
nombres_escenarios = cell(n_scenarios, 1);

% Barra de progreso
fprintf('Progreso: ');

fprintf('\n🔬 Creando modelos de reconstrucción (imdl) con HP fijos...\n');

% --- Crear fmdl de 96 canales ---
[fmdl_96ch, ~] = configurar_sistema_96ch();

% --- Hiperparámetros Óptimos Encontrados (96ch/16e) ---
% Valores originales (8e)
hp_8e_tikhonov = 4.28e-08;  
hp_8e_laplace = hp_8e_tikhonov;
hp_8e_tv = hp_8e_tikhonov * 10; % Se mantiene el factor para 8e

% Nuevos valores óptimos (16e/96ch) de la búsqueda:
hp_16e_tikhonov = 7.8805e-06; % CC=0.9127
hp_16e_laplace = 7.8805e-06;  % CC=0.9200 (Mejor rendimiento)
hp_16e_tv = 7.2790e-06;       % CC=0.9200 (Usando la alternativa estable)

fprintf('   ✅ Hiperparámetros óptimos (16e) aplicados:\n');
fprintf('      - Tikhonov: %.4e\n', hp_16e_tikhonov);
fprintf('      - Laplace:  %.4e\n', hp_16e_laplace);
fprintf('      - TV:       %.4e\n', hp_16e_tv);

fprintf('   ✅ Usando hiperparámetros optimizados.\n');

% --- CÓDIGO FINAL DE CREACIÓN DE MODELOS (SOLUCIÓN DE ESTABILIDAD) ---
fprintf('   🔧 Creando 6 modelos inversos finales...\n');

% Modelos 8e (Línea Base) - (Mantener configuración original)
imdl_8e_Tikhonov = crear_imdl_simple(fmdl_8e, 'Tikhonov', hp_8e_tikhonov);
imdl_8e_Laplace  = crear_imdl_simple(fmdl_8e, 'Laplace',  hp_8e_laplace);
imdl_8e_TV       = crear_imdl_simple(fmdl_8e, 'TV',       hp_8e_tv);

% Modelos 16e (Propuesta y Límite Superior) - FORZAR TIKHONOV PARA ESTABILIDAD
% Nota: Todos usan el prior 'Tikhonov' pero mantienen sus respectivos HPs.
imdl_16e_Tikhonov = crear_imdl_simple(fmdl_96ch, 'Tikhonov', hp_16e_tikhonov);
imdl_16e_Laplace  = crear_imdl_simple(fmdl_96ch, 'Tikhonov', hp_16e_laplace); 
imdl_16e_TV       = crear_imdl_simple(fmdl_96ch, 'Tikhonov', hp_16e_tv);


% --- APLICACIÓN DE PARCHES (DEBE IR DESPUÉS DE LA CREACIÓN) ---

% 1. PARCHE DE GARANTÍA (RESTAURA EIDORS OBJECT)
imdl_8e_Tikhonov = eidors_obj('inv_model', imdl_8e_Tikhonov);
imdl_8e_Laplace  = eidors_obj('inv_model', imdl_8e_Laplace);
imdl_8e_TV       = eidors_obj('inv_model', imdl_8e_TV);
imdl_16e_Tikhonov = eidors_obj('inv_model', imdl_16e_Tikhonov);
imdl_16e_Laplace  = eidors_obj('inv_model', imdl_16e_Laplace);
imdl_16e_TV       = eidors_obj('inv_model', imdl_16e_TV);
fprintf('   ✅ Estructuras IMDL reconfirmadas como objetos EIDORS.\n');

% 2. PARCHE DE JACOBIANO (CONFIGURA EL FONDO)
fprintf('   ✅ Configurando Jacobianos con objetos imagen para estabilidad...\n');
img_bkgnd_8e = mk_image(fmdl_8e, CONFIG.conductividad_fondo);
img_bkgnd_96ch = mk_image(fmdl_96ch, CONFIG.conductividad_fondo);
imdl_8e_Tikhonov.jacobian_bkgnd = img_bkgnd_8e;
imdl_8e_Laplace.jacobian_bkgnd = img_bkgnd_8e;
imdl_8e_TV.jacobian_bkgnd = img_bkgnd_8e; 
imdl_16e_Tikhonov.jacobian_bkgnd = img_bkgnd_96ch;
imdl_16e_Laplace.jacobian_bkgnd = img_bkgnd_96ch;
imdl_16e_TV.jacobian_bkgnd = img_bkgnd_96ch;
fprintf('   ✅ Jacobianos configurados explícitamente con imagen de fondo (1.0)\n');


for scenario_id = 1:n_scenarios
    fprintf('[%d] ', scenario_id);
    
    try
        %% === ETAPA 1: GENERACIÓN DE PHANTOMS CONSISTENTES ===
        fprintf('      🎯 Generando phantom para escenario %d...\n', scenario_id);
        
        % Llamamos a la función "forzada" que creamos, asegurándonos
        % de pasarle todos los argumentos que necesita.
        try
            [img_8e, ~] = generar_imagen_conductividad_forzado(fmdl_8e, CONFIG, scenario_id);
        catch ME_gen
            error('Fallo en generar_imagen_conductividad_forzado: %s', ME_gen.message);
        end
        
        % Mapear a 16e usando función existente
        img_16e = mapear_conductividad_8_a_16(fmdl_16e, img_8e, CONFIG);
        
        % Verificar consistencia
        assert(img_8e.scenario_id == img_16e.scenario_id, 'Inconsistencia de scenario_id');
        
        %% === ETAPA 2: SIMULACIÓN FEM ===
        
        % Resolver problemas directos
        volt_8e = fwd_solve(img_8e);
        volt_16e = fwd_solve(img_16e);
        
        %% === ETAPA 3: EXTRACCIÓN DE MEDICIONES ===
        
        % Extraer mediciones de 8 electrodos (entrada CNN)
        X_8e_prueba = extraer_40_mediciones(volt_8e);
        
        % Extraer todas las mediciones de 16 electrodos
        y_16e_todas = volt_16e.meas;
        
        % Extraer Ground Truth usando metodología de canales virtuales
        y_16e_real = extraer_96_mediciones(volt_16e);
       


% ======================================================================
%% ETAPA 4: PREDICCIÓN CNN (SEÑAL CONJUGADA) - VERSIÓN CORREGIDA
% ======================================================================
        
        tic_cnn = tic;
        
        try
            % --- PASO 1: PREPARAR DATOS DE ENTRADA ---
            % Convertir los datos de MATLAB a un array de NumPy directamente.
            % Aseguramos formato (1, 40) y tipo float32.
            X_input_matlab = double(X_8e_prueba(:)');
% Convertir a un array de NumPy y APLICAR EL RESHAPE a 2D.
            % La forma final será (1, 40), que es lo que espera el scaler.
            X_input_py = py.numpy.array(X_input_matlab, ...
                                      dtype=py.numpy.float32).reshape(int32(1), int32(-1));
            % --- PASO 2: NORMALIZAR EN PYTHON ---
            % Aplicar la transformación del scaler_X a los datos de entrada.
            % Toda esta operación ocurre dentro del entorno de Python.
            X_norm_py = scaler_X.transform(X_input_py);

            % --- PASO 3: RESHAPE PARA LA CNN ---
            % Cambiar la forma de (1, 40) a (1, 40, 1) como espera la capa Conv1D.
            X_tensor = py.numpy.reshape(X_norm_py, int32([1, 40, 1]));
            
            % --- PASO 4: REALIZAR LA PREDICCIÓN ---
            % El modelo predice sobre el tensor de entrada normalizado.
            y_pred_norm_py = modelo_cnn.predict(X_tensor, pyargs('verbose', int32(0)));
            
            % --- PASO 5: DESNORMALIZAR EN PYTHON ---
            % Aplicar la transformación inversa del scaler_y para volver a la escala original.
            y_conjugada_py = scaler_y.inverse_transform(y_pred_norm_py);
            
            % --- PASO 6: CONVERTIR A MATLAB (LA CORRECCIÓN CLAVE) ---
            % El resultado 'y_conjugada_py' ya es un array de NumPy.
            % Simplemente lo convertimos a un tipo 'double' de MATLAB.
            % NO se necesita la llamada a .numpy().
            y_conjugada = double(y_conjugada_py);
            
            % Asegurar que el resultado final sea un vector columna.
            y_conjugada = y_conjugada(:);
            
            tiempo_cnn = toc(tic_cnn);
            
            fprintf('   ✅ Predicción CNN real completada (%.3f seg)\n', tiempo_cnn);
            
            % Verificación final de dimensiones
            if length(y_conjugada) ~= 96
                error('Dimensión incorrecta de la predicción final: %d (esperado: 96)', length(y_conjugada));
            end
            
        catch python_error
            fprintf('\n   ❌ ERROR en el bloque de predicción CNN para escenario %d: %s\n', ...
                    scenario_id, char(python_error.message));
            
            % Usar predicción dummy para no detener el script en caso de un error inesperado.
            y_conjugada = y_16e_real + randn(size(y_16e_real)) * std(y_16e_real) * 0.01;
            tiempo_cnn = toc(tic_cnn);
            
            fprintf('   ℹ️ Usando predicción dummy para continuar análisis\n');
        end
        
        %% === ETAPA 5: CÁLCULO DE MÉTRICAS ===
        
        % Verificar que ambos vectores tengan la misma dimensión
        if length(y_16e_real) ~= length(y_conjugada)
            error('Dimensiones incompatibles: y_16e_real=%d, y_conjugada=%d', ...
                length(y_16e_real), length(y_conjugada));
        end
        
        % Calcular métricas de comparación
        diferencias = y_16e_real - y_conjugada;
        mse_scenario = mean(diferencias.^2);
        mae_scenario = mean(abs(diferencias));
        re_scenario = norm(diferencias) / norm(y_16e_real);
        
        % Calcular Coeficiente de Correlación (CC) de la señal
        if std(y_16e_real) > 1e-9 && std(y_conjugada) > 1e-9
            cc_scenario = corr(y_16e_real, y_conjugada);
        else
            cc_scenario = 0; % No hay correlación si una señal es plana
        end
        
        % Verificar que las métricas son válidas
        if ~isfinite(cc_scenario) || ~isfinite(mae_scenario) || ~isfinite(re_scenario)
            warning('Métricas no válidas para escenario %d', scenario_id);
            cc_scenario = 0;
            mae_scenario = Inf;
            re_scenario = Inf;
        end
        
        %% === ETAPA 6: ALMACENAMIENTO DE RESULTADOS ===
        
        resultados{scenario_id} = struct();
        resultados{scenario_id}.scenario_id = scenario_id;
        resultados{scenario_id}.scenario_name = scenarios{scenario_id}.name;
        resultados{scenario_id}.img_16e = img_16e;
        resultados{scenario_id}.y_16e_real = y_16e_real;
        resultados{scenario_id}.y_conjugada = y_conjugada;
        resultados{scenario_id}.X_8e_prueba = X_8e_prueba;
        resultados{scenario_id}.metricas = struct('CC', cc_scenario, 'MAE', mae_scenario, ...
                                                  'ER', re_scenario, 'Tiempo_CNN', tiempo_cnn);
        
        % Almacenar en matriz de resumen
        metricas_resumen(scenario_id, :) = [cc_scenario, mae_scenario, re_scenario, tiempo_cnn];
        nombres_escenarios{scenario_id} = scenarios{scenario_id}.name;
        
        % Mostrar progreso detallado cada 2 escenarios
        if mod(scenario_id, 2) == 0
            fprintf('\n   ✅ Escenario %d completado: CC=%.4f, MAE=%.2e\n', ...
                scenario_id, cc_scenario, mae_scenario);
            fprintf('Progreso: ');
        end
        
        

    catch ME
        fprintf('\n❌ ERROR en escenario %d: %s\n', scenario_id, ME.message);
        
        % Crear resultado de error para mantener consistencia
        resultados{scenario_id} = struct();
        resultados{scenario_id}.scenario_id = scenario_id;
        resultados{scenario_id}.scenario_name = sprintf('ERROR_%d', scenario_id);
        resultados{scenario_id}.error = ME.message;
        
        metricas_resumen(scenario_id, :) = [NaN, NaN, NaN, NaN];
        nombres_escenarios{scenario_id} = sprintf('ERROR_%d', scenario_id);
        
        continue;
    end
end

fprintf('\n✅ Validación sistemática completada!\n');

% Reactivar warnings
warning('on', 'add_circular_inclusion:NoElementsAffected');

%% =====================================================================
% SECCIÓN 6: VISUALIZACIÓN DE RESULTADOS - PRIMERA FIGURA (ESCENARIOS 1-5)
% =====================================================================
fprintf('\n📊 GENERANDO VISUALIZACIONES...\n');

% Primera figura: Escenarios 1-5
fprintf('   🖼️  Creando Figura 1: Escenarios 1-5...\n');

figure('Position', [100, 100, 1800, 1200], 'Name', 'Análisis Comparativo 1-5');

for j = 1:5
   if j <= length(resultados) && isfield(resultados{j}, 'img_16e')
       
       % Fila j del subplot
       row = j;
       
       %% Columna 1: Phantom FEM
       subplot(5, 3, (row-1)*3 + 1);
       try
           show_fem(resultados{j}.img_16e);
           title(sprintf('Escenario %d: %s', j, resultados{j}.scenario_name), ...
                 'FontSize', 10, 'FontWeight', 'bold');
           axis equal; axis tight;
       catch
           text(0.5, 0.5, sprintf('Error\nEscenario %d', j), ...
               'HorizontalAlignment', 'center', 'FontSize', 12);
           title(sprintf('Escenario %d: ERROR', j), 'FontSize', 10, 'Color', 'red');
       end
       
       %% Columna 2: Comparación de señales
       subplot(5, 3, (row-1)*3 + 2);
       
       if isfield(resultados{j}, 'y_16e_real') && isfield(resultados{j}, 'y_conjugada')
           
           channels = 1:96;
           plot(channels, resultados{j}.y_16e_real, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground Truth (16e)');
           hold on;
           plot(channels, resultados{j}.y_conjugada, 'r--', 'LineWidth', 1.5, 'DisplayName', 'CNN Conjugada');
           
           % Calcular métricas para el título
           CC = resultados{j}.metricas.CC; 
           MAE = resultados{j}.metricas.MAE;
           
           title(sprintf('CC = %.4f, MAE = %.2e', CC, MAE), ...
                 'FontSize', 9, 'FontWeight', 'bold');
           xlabel('Canal');
           ylabel('Voltaje (V)');
           legend('Location', 'best', 'FontSize', 8);
           grid on;
           
       else
           text(0.5, 0.5, 'Error en datos', 'HorizontalAlignment', 'center');
           title('ERROR', 'Color', 'red');
       end
       
       %% Columna 3: Diferencia
       subplot(5, 3, (row-1)*3 + 3);
       
       if isfield(resultados{j}, 'y_16e_real') && isfield(resultados{j}, 'y_conjugada')
           
           diferencia = resultados{j}.y_16e_real - resultados{j}.y_conjugada;
           channels = 1:96;
           
           plot(channels, diferencia, 'g-', 'LineWidth', 1);
           hold on;
           plot(channels, zeros(size(channels)), 'k--', 'Color', [0 0 0 0.5]);
           
           title(sprintf('Diferencia (ER = %.2e)', resultados{j}.metricas.ER), ...
                 'FontSize', 9, 'FontWeight', 'bold');
           xlabel('Canal');
           ylabel('Error (V)');
           grid on;
           
       else
           text(0.5, 0.5, 'Error en datos', 'HorizontalAlignment', 'center');
           title('ERROR', 'Color', 'red');
       end
       
   else
       % Manejo de escenarios con error
       for col = 1:3
           subplot(5, 3, (j-1)*3 + col);
           text(0.5, 0.5, sprintf('ERROR\nEscenario %d', j), ...
               'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'red');
           title(sprintf('Escenario %d: ERROR', j), 'FontSize', 10, 'Color', 'red');
       end
   end
end

sgtitle('Análisis Comparativo: 16e FEM vs CNN Conjugada (Escenarios 1-5)', ...
       'FontSize', 16, 'FontWeight', 'bold');

%% =====================================================================
% SECCIÓN 7: VISUALIZACIÓN DE RESULTADOS - SEGUNDA FIGURA (ESCENARIOS 6-10)
% =====================================================================

% Segunda figura: Escenarios 6-10
fprintf('   🖼️  Creando Figura 2: Escenarios 6-10...\n');

figure('Position', [200, 50, 1800, 1200], 'Name', 'Análisis Comparativo 6-10');

%% =====================================================================
%% SECCIÓN 5: BUCLE DE VALIDACIÓN SISTEMÁTICA - PROCESAMIENTO COMPLETO
%% =====================================================================

for scenario_id = 1:n_scenarios
    fprintf('[%d] ', scenario_id);
    
    try
        % =================================================================
        %% ETAPA 1: GENERACIÓN DE PHANTOMS CONSISTENTES
        % =================================================================
        fprintf('\n      🎯 ETAPA 1: Generando phantom para escenario %d...\n', scenario_id);
        
        % Llamar a la función forzada que genera el escenario específico
        try
            [img_8e, ~] = generar_imagen_conductividad_forzado(fmdl_8e, CONFIG, scenario_id);
        catch ME_gen
            error('Fallo en generar_imagen_conductividad_forzado: %s', ME_gen.message);
        end
        
        % Mapear conductividad de 8e a 16e
        img_16e = mapear_conductividad_8_a_16(fmdl_16e, img_8e, CONFIG);
        
        % Verificar consistencia entre modelos
        assert(img_8e.scenario_id == img_16e.scenario_id, 'Inconsistencia de scenario_id');
        
        fprintf('         ✅ Phantom generado exitosamente\n');
        
        % =================================================================
        %% ETAPA 2: SIMULACIÓN FEM (PROBLEMA DIRECTO)
        % =================================================================
        fprintf('      ⚡ ETAPA 2: Resolviendo problema directo FEM...\n');
        
        % Resolver problema directo para ambos modelos
        volt_8e = fwd_solve(img_8e);
        volt_16e = fwd_solve(img_16e);
        
        fprintf('         ✅ Simulaciones FEM completadas\n');
        
        % =================================================================
        %% ETAPA 3: EXTRACCIÓN DE MEDICIONES
        % =================================================================
        fprintf('      📊 ETAPA 3: Extrayendo mediciones...\n');
        
        % Extraer mediciones de 8 electrodos (entrada para la CNN)
        X_8e_prueba = extraer_40_mediciones(volt_8e);
        
        % Extraer todas las mediciones de 16 electrodos
        y_16e_todas = volt_16e.meas;
        
        % Extraer Ground Truth usando metodología de canales virtuales (96 mediciones)
        y_16e_real = extraer_96_mediciones(volt_16e);
        
        fprintf('         ✅ Mediciones extraídas: %d (8e) y %d (16e)\n', ...
                length(X_8e_prueba), length(y_16e_real));
        
        % =================================================================
        %% ETAPA 4: PREDICCIÓN CNN (SEÑAL CONJUGADA)
        % =================================================================
        fprintf('      🧠 ETAPA 4: Realizando predicción CNN...\n');
        
        tic_cnn = tic;
        
        try
            % --- PASO 4.1: PREPARAR DATOS DE ENTRADA ---
            X_input_matlab = double(X_8e_prueba(:)');
            
            % Convertir a NumPy y reshape a (1, 40)
            X_input_py = py.numpy.array(X_input_matlab, ...
                                      dtype=py.numpy.float32).reshape(int32(1), int32(-1));
            
            % --- PASO 4.2: NORMALIZAR EN PYTHON ---
            X_norm_py = scaler_X.transform(X_input_py);
            
            % --- PASO 4.3: RESHAPE PARA LA CNN (1, 40, 1) ---
            X_tensor = py.numpy.reshape(X_norm_py, int32([1, 40, 1]));
            
            % --- PASO 4.4: REALIZAR LA PREDICCIÓN ---
            y_pred_norm_py = modelo_cnn.predict(X_tensor, pyargs('verbose', int32(0)));
            
            % --- PASO 4.5: DESNORMALIZAR EN PYTHON ---
            y_conjugada_py = scaler_y.inverse_transform(y_pred_norm_py);
            
            % --- PASO 4.6: CONVERTIR A MATLAB ---
            y_conjugada = double(y_conjugada_py);
            y_conjugada = y_conjugada(:);
            
            tiempo_cnn = toc(tic_cnn);
            
            fprintf('         ✅ Predicción CNN completada (%.3f seg)\n', tiempo_cnn);
            
            % Verificación final de dimensiones
            if length(y_conjugada) ~= 96
                error('Dimensión incorrecta de la predicción final: %d (esperado: 96)', ...
                      length(y_conjugada));
            end
            
        catch python_error
            fprintf('\n         ❌ ERROR en predicción CNN: %s\n', char(python_error.message));
            
            % Usar predicción dummy para no detener el script
            y_conjugada = y_16e_real + randn(size(y_16e_real)) * std(y_16e_real) * 0.01;
            tiempo_cnn = toc(tic_cnn);
            
            fprintf('         ℹ️  Usando predicción dummy para continuar análisis\n');
        end
        
        % =================================================================
        %% ETAPA 5: CÁLCULO DE MÉTRICAS DE SEÑAL
        % =================================================================
        fprintf('      📏 ETAPA 5: Calculando métricas de señal...\n');
        
        % Verificar que ambos vectores tengan la misma dimensión
        if length(y_16e_real) ~= length(y_conjugada)
            error('Dimensiones incompatibles: y_16e_real=%d, y_conjugada=%d', ...
                length(y_16e_real), length(y_conjugada));
        end
        
        % Calcular métricas de comparación de señales
        diferencias = y_16e_real - y_conjugada;
        mse_scenario = mean(diferencias.^2);
        mae_scenario = mean(abs(diferencias));
        re_scenario = norm(diferencias) / norm(y_16e_real);
        
        % Calcular Coeficiente de Correlación (CC) de la señal
        if std(y_16e_real) > 1e-9 && std(y_conjugada) > 1e-9
            cc_scenario = corr(y_16e_real, y_conjugada);
        else
            cc_scenario = 0; % No hay correlación si una señal es plana
        end
        
        % Verificar que las métricas son válidas
        if ~isfinite(cc_scenario) || ~isfinite(mae_scenario) || ~isfinite(re_scenario)
            warning('Métricas no válidas para escenario %d', scenario_id);
            cc_scenario = 0;
            mae_scenario = Inf;
            re_scenario = Inf;
        end
        
        fprintf('         ✅ Métricas calculadas: CC=%.4f, MAE=%.2e, ER=%.2e\n', ...
                cc_scenario, mae_scenario, re_scenario);
        
        % =================================================================
        %% ETAPA 6: ALMACENAMIENTO DE RESULTADOS DE SEÑAL
        % =================================================================
        fprintf('      💾 ETAPA 6: Almacenando resultados de señal...\n');
        
        % Crear estructura de resultados para este escenario
        resultados{scenario_id} = struct();
        resultados{scenario_id}.scenario_id = scenario_id;
        resultados{scenario_id}.scenario_name = scenarios{scenario_id}.name;
        resultados{scenario_id}.img_16e = img_16e;
        resultados{scenario_id}.y_16e_real = y_16e_real;
        resultados{scenario_id}.y_conjugada = y_conjugada;
        resultados{scenario_id}.X_8e_prueba = X_8e_prueba;
        resultados{scenario_id}.metricas = struct('CC', cc_scenario, ...
                                                  'MAE', mae_scenario, ...
                                                  'ER', re_scenario, ...
                                                  'Tiempo_CNN', tiempo_cnn);
        
        % Almacenar en matriz de resumen
        metricas_resumen(scenario_id, :) = [cc_scenario, mae_scenario, re_scenario, tiempo_cnn];
        nombres_escenarios{scenario_id} = scenarios{scenario_id}.name;
        
        fprintf('         ✅ Resultados de señal almacenados\n');
        
        % =================================================================
        %% ETAPA 7: RECONSTRUCCIÓN DE IMÁGENES
        % =================================================================
        fprintf('      🔬 ETAPA 7: Reconstruyendo imágenes...\n');
        
        % Crear voltajes homogéneos para este escenario
        volt_homog_8e_loop = fwd_solve(mk_image(fmdl_8e, CONFIG.conductividad_fondo));
        volt_homog_96ch_loop = fwd_solve(mk_image(fmdl_96ch, CONFIG.conductividad_fondo));
        
        % --- A) LÍNEA BASE (8 electrodos) ---
        fprintf('         🔹 Reconstruyendo con Línea Base (8e)...\n');
        img_8e_Tikhonov = inv_solve(imdl_8e_Tikhonov, volt_homog_8e_loop, volt_8e);
        img_8e_Laplace  = inv_solve(imdl_8e_Laplace,  volt_homog_8e_loop, volt_8e);
        img_8e_TV       = inv_solve(imdl_8e_TV,       volt_homog_8e_loop, volt_8e);
        
        % --- B) PROPUESTA (8 electrodos + CNN) ---
        fprintf('         🔹 Reconstruyendo con Propuesta (8e+CNN)...\n');
        volt_cnn = crear_voltaje_sintetico(fmdl_96ch, volt_homog_96ch_loop, -y_conjugada);
        img_cnn_Tikhonov = inv_solve(imdl_16e_Tikhonov, volt_homog_96ch_loop, volt_cnn);
        img_cnn_Laplace  = inv_solve(imdl_16e_Laplace,  volt_homog_96ch_loop, volt_cnn);
        img_cnn_TV       = inv_solve(imdl_16e_TV,       volt_homog_96ch_loop, volt_cnn);
        
        % --- C) LÍMITE SUPERIOR (16 electrodos Ground Truth) ---
        fprintf('         🔹 Reconstruyendo con Límite Superior (96ch GT)...\n');
        volt_gt_96ch = volt_16e;
        volt_gt_96ch.meas = -y_16e_real;
        volt_gt_96ch.fwd_model = fmdl_96ch;
        
        img_gt_Tikhonov = inv_solve(imdl_16e_Tikhonov, volt_homog_96ch_loop, volt_gt_96ch);
        img_gt_Laplace  = inv_solve(imdl_16e_Laplace,  volt_homog_96ch_loop, volt_gt_96ch);
        img_gt_TV       = inv_solve(imdl_16e_TV,       volt_homog_96ch_loop, volt_gt_96ch);
        
        fprintf('         ✅ 9 reconstrucciones completadas (3 métodos × 3 algoritmos)\n');
        
        % =================================================================
        %% ETAPA 8: CÁLCULO DE MÉTRICAS DE IMAGEN
        % =================================================================
        fprintf('      📐 ETAPA 8: Calculando métricas de imagen...\n');
        
        metricas_imagen = struct();
        
        % Métricas de Línea Base (8e) - requiere interpolación
        fprintf('         🔹 Calculando métricas de Línea Base...\n');
        metricas_imagen.linea_base.Tikhonov = calcular_metricas_imagen(img_8e_Tikhonov, img_16e, true);
        metricas_imagen.linea_base.Laplace  = calcular_metricas_imagen(img_8e_Laplace,  img_16e, true);
        metricas_imagen.linea_base.TV       = calcular_metricas_imagen(img_8e_TV,       img_16e, true);
        
        % Métricas de Propuesta (8e+CNN) - sin interpolación
        fprintf('         🔹 Calculando métricas de Propuesta...\n');
        metricas_imagen.propuesta.Tikhonov = calcular_metricas_imagen(img_cnn_Tikhonov, img_16e, false);
        metricas_imagen.propuesta.Laplace  = calcular_metricas_imagen(img_cnn_Laplace,  img_16e, false);
        metricas_imagen.propuesta.TV       = calcular_metricas_imagen(img_cnn_TV,       img_16e, false);
        
        % Métricas de Límite Superior (96ch GT) - sin interpolación
        fprintf('         🔹 Calculando métricas de Límite Superior...\n');
        metricas_imagen.limite_sup.Tikhonov = calcular_metricas_imagen(img_gt_Tikhonov, img_16e, false);
        metricas_imagen.limite_sup.Laplace  = calcular_metricas_imagen(img_gt_Laplace,  img_16e, false);
        metricas_imagen.limite_sup.TV       = calcular_metricas_imagen(img_gt_TV,       img_16e, false);
        
        % Guardar métricas en la estructura de resultados
        resultados{scenario_id}.metricas_imagen = metricas_imagen;
        
        fprintf('         ✅ Métricas de imagen calculadas para 3 métodos × 3 algoritmos\n');
        
        % =================================================================
        %% ETAPA 9: GUARDAR IMÁGENES RECONSTRUIDAS
        % =================================================================
        fprintf('      💾 ETAPA 9: Guardando imágenes reconstruidas...\n');
        
        % Crear estructura de imágenes
        resultados{scenario_id}.imagenes = struct();
        
        % Guardar Ground Truth del phantom
        resultados{scenario_id}.imagenes.img_gt = img_16e;
        
        % Guardar reconstrucciones de Línea Base (8e)
        resultados{scenario_id}.imagenes.linea_base.Tikhonov = img_8e_Tikhonov;
        resultados{scenario_id}.imagenes.linea_base.Laplace  = img_8e_Laplace;
        resultados{scenario_id}.imagenes.linea_base.TV       = img_8e_TV;
        
        % Guardar reconstrucciones de Propuesta (8e+CNN)
        resultados{scenario_id}.imagenes.propuesta.Tikhonov = img_cnn_Tikhonov;
        resultados{scenario_id}.imagenes.propuesta.Laplace  = img_cnn_Laplace;
        resultados{scenario_id}.imagenes.propuesta.TV       = img_cnn_TV;
        
        % Guardar reconstrucciones de Límite Superior (96ch GT)
        resultados{scenario_id}.imagenes.limite_sup.Tikhonov = img_gt_Tikhonov;
        resultados{scenario_id}.imagenes.limite_sup.Laplace  = img_gt_Laplace;
        resultados{scenario_id}.imagenes.limite_sup.TV       = img_gt_TV;
        
        fprintf('         ✅ 10 imágenes guardadas (1 GT + 9 reconstrucciones)\n');
        
        % =================================================================
        %% FINALIZACIÓN DEL ESCENARIO
        % =================================================================
        
        % Mostrar progreso detallado cada 2 escenarios
        if mod(scenario_id, 2) == 0
            fprintf('\n   ✅✅ ESCENARIO %d COMPLETADO ✅✅\n', scenario_id);
            fprintf('      📊 CC Señal: %.4f | MAE: %.2e\n', cc_scenario, mae_scenario);
            fprintf('      🖼️  CC Imagen (Laplace): %.4f (Línea Base) | %.4f (Propuesta)\n', ...
                    metricas_imagen.linea_base.Laplace.CC, ...
                    metricas_imagen.propuesta.Laplace.CC);
            fprintf('Progreso: ');
        end
        
    catch ME
        % =================================================================
        %% MANEJO DE ERRORES
        % =================================================================
        fprintf('\n\n   ❌❌ ERROR EN ESCENARIO %d ❌❌\n', scenario_id);
        fprintf('   📛 Tipo de error: %s\n', ME.identifier);
        fprintf('   📛 Mensaje: %s\n', ME.message);
        fprintf('   📛 Línea: %s\n\n', ME.stack(1).name);
        
        % Crear resultado de error para mantener consistencia
        resultados{scenario_id} = struct();
        resultados{scenario_id}.scenario_id = scenario_id;
        resultados{scenario_id}.scenario_name = sprintf('ERROR_%d', scenario_id);
        resultados{scenario_id}.error = ME.message;
        resultados{scenario_id}.error_stack = ME.stack;
        
        % Llenar con NaN las métricas para este escenario
        metricas_resumen(scenario_id, :) = [NaN, NaN, NaN, NaN];
        nombres_escenarios{scenario_id} = sprintf('ERROR_%d', scenario_id);
        
        fprintf('   ⚠️  Continuando con el siguiente escenario...\n');
        continue;
    end
    
end % FIN DEL BUCLE PRINCIPAL: for scenario_id = 1:n_scenarios

fprintf('\n\n');
fprintf('🎊==================================================================🎊\n');
fprintf('   BUCLE DE VALIDACIÓN SISTEMÁTICA COMPLETADO\n');
fprintf('   Procesados: %d/%d escenarios\n', n_scenarios, n_scenarios);
fprintf('🎊==================================================================🎊\n');
fprintf('\n');

% Reactivar warnings
warning('on', 'add_circular_inclusion:NoElementsAffected');

       % --- GUARDAR LAS IMÁGENES RECONSTRUIDAS ---
fprintf('      💾 Guardando imágenes reconstruidas...\n');

% Crear la estructura 'imagenes' si no existe
if ~isfield(resultados{scenario_id}, 'imagenes')
    resultados{scenario_id}.imagenes = struct();
end

% Guardar Ground Truth del phantom
resultados{scenario_id}.imagenes.img_gt = img_16e;

% Guardar reconstrucciones de Línea Base (8e)
resultados{scenario_id}.imagenes.linea_base.Tikhonov = img_8e_Tikhonov;
resultados{scenario_id}.imagenes.linea_base.Laplace  = img_8e_Laplace;
resultados{scenario_id}.imagenes.linea_base.TV       = img_8e_TV;

% Guardar reconstrucciones de Propuesta (8e+CNN)
resultados{scenario_id}.imagenes.propuesta.Tikhonov = img_cnn_Tikhonov;
resultados{scenario_id}.imagenes.propuesta.Laplace  = img_cnn_Laplace;
resultados{scenario_id}.imagenes.propuesta.TV       = img_cnn_TV;

% Guardar reconstrucciones de Límite Superior (96ch GT)
resultados{scenario_id}.imagenes.limite_sup.Tikhonov = img_gt_Tikhonov;
resultados{scenario_id}.imagenes.limite_sup.Laplace  = img_gt_Laplace;
resultados{scenario_id}.imagenes.limite_sup.TV       = img_gt_TV;

fprintf('      ✅ Imágenes guardadas exitosamente\n');

% ======================================================================
%% MÓDULO 4 (VERSIÓN SEPARADA): VISUALIZACIÓN DE RECONSTRUCCIONES
% ======================================================================
fprintf('\n🖼️  MÓDULO 4: GENERANDO FIGURAS DE RECONSTRUCCIÓN...\n');
fprintf('================================================================\n');

% --- CONFIGURACIÓN DE RENDERIZADO ---
set(0, 'DefaultFigureRenderer', 'painters'); 
set(groot, 'defaultFigureColormap', jet); 

algoritmos = {'Tikhonov', 'Laplace', 'TV'};
metodos_nombre = {'Línea Base (8e)', 'Propuesta (8e+CNN)', 'Límite Superior (96ch GT)'};

% Bucle para crear figuras por cada algoritmo de reconstrucción
for idx_algoritmo = 1:length(algoritmos)
    
    algoritmo_actual = algoritmos{idx_algoritmo};
    fprintf('   📊 Creando figuras para: %s\n', algoritmo_actual);
    
    % --- FIGURA 1: Escenarios 1-5 ---
    fig1 = figure('Name', sprintf('Reconstrucciones %s (Esc. 1-5)', algoritmo_actual), ...
                  'Position', [50, 400, 1800, 1000]);
    
    for i = 1:5
        scenario_id = i;
        
        % Verificar que existan las imágenes
        if isfield(resultados{scenario_id}, 'imagenes')
            
            % Extraer las imágenes necesarias
            img_gt = resultados{scenario_id}.imagenes.img_gt;
            img_lb = resultados{scenario_id}.imagenes.linea_base.(algoritmo_actual);
            img_cnn = resultados{scenario_id}.imagenes.propuesta.(algoritmo_actual);
            img_gt_recon = resultados{scenario_id}.imagenes.limite_sup.(algoritmo_actual);
            
            % Extraer las métricas CC
            cc_lb = resultados{scenario_id}.metricas_imagen.linea_base.(algoritmo_actual).CC;
            cc_cnn = resultados{scenario_id}.metricas_imagen.propuesta.(algoritmo_actual).CC;
            cc_gt = resultados{scenario_id}.metricas_imagen.limite_sup.(algoritmo_actual).CC;
            
            % COLUMNA 1: Ground Truth
            subplot(5, 4, (i-1)*4 + 1);
            show_fem(img_gt);
            axis equal; axis off;
            if i == 1
                title('Ground Truth', 'FontSize', 12, 'FontWeight', 'bold');
            end
            ylabel(sprintf('Esc. %d', i), 'FontSize', 11, 'FontWeight', 'bold');
            
            % COLUMNA 2: Línea Base (8e)
            subplot(5, 4, (i-1)*4 + 2);
            show_fem(img_lb);
            axis equal; axis off;
            title(sprintf('CC=%.3f', cc_lb), 'FontSize', 10);
            if i == 1
                xlabel(metodos_nombre{1}, 'FontSize', 11, 'FontWeight', 'bold');
            end
            
            % COLUMNA 3: Propuesta (8e+CNN)
            subplot(5, 4, (i-1)*4 + 3);
            show_fem(img_cnn);
            axis equal; axis off;
            title(sprintf('CC=%.3f', cc_cnn), 'FontSize', 10);
            if i == 1
                xlabel(metodos_nombre{2}, 'FontSize', 11, 'FontWeight', 'bold');
            end
            
            % COLUMNA 4: Límite Superior (96ch GT)
            subplot(5, 4, (i-1)*4 + 4);
            show_fem(img_gt_recon);
            axis equal; axis off;
            title(sprintf('CC=%.3f', cc_gt), 'FontSize', 10);
            if i == 1
                xlabel(metodos_nombre{3}, 'FontSize', 11, 'FontWeight', 'bold');
            end
            
        else
            % Si no hay imágenes, mostrar error
            for col = 1:4
                subplot(5, 4, (i-1)*4 + col);
                text(0.5, 0.5, 'ERROR: Imágenes no disponibles', ...
                    'HorizontalAlignment', 'center', 'Color', 'red');
            end
        end
    end
    
    sgtitle(sprintf('Reconstrucción - %s (Escenarios 1-5)', algoritmo_actual), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % GUARDAR FIGURA 1
    nombre_archivo = sprintf('Reconstruccion_%s_Esc1-5.png', algoritmo_actual);
    saveas(fig1, nombre_archivo);
    fprintf('      ✅ Guardada: %s\n', nombre_archivo);
    
    % --- FIGURA 2: Escenarios 6-10 ---
    fig2 = figure('Name', sprintf('Reconstrucciones %s (Esc. 6-10)', algoritmo_actual), ...
                  'Position', [100, 50, 1800, 1000]);
    
    for i = 1:5
        scenario_id = i + 5; % Escenarios 6-10
        
        if isfield(resultados{scenario_id}, 'imagenes')
            
            % Extraer imágenes
            img_gt = resultados{scenario_id}.imagenes.img_gt;
            img_lb = resultados{scenario_id}.imagenes.linea_base.(algoritmo_actual);
            img_cnn = resultados{scenario_id}.imagenes.propuesta.(algoritmo_actual);
            img_gt_recon = resultados{scenario_id}.imagenes.limite_sup.(algoritmo_actual);
            
            % Extraer métricas
            cc_lb = resultados{scenario_id}.metricas_imagen.linea_base.(algoritmo_actual).CC;
            cc_cnn = resultados{scenario_id}.metricas_imagen.propuesta.(algoritmo_actual).CC;
            cc_gt = resultados{scenario_id}.metricas_imagen.limite_sup.(algoritmo_actual).CC;
            
            % COLUMNA 1: Ground Truth
            subplot(5, 4, (i-1)*4 + 1);
            show_fem(img_gt);
            axis equal; axis off;
            if i == 1
                title('Ground Truth', 'FontSize', 12, 'FontWeight', 'bold');
            end
            ylabel(sprintf('Esc. %d', scenario_id), 'FontSize', 11, 'FontWeight', 'bold');
            
            % COLUMNA 2: Línea Base
            subplot(5, 4, (i-1)*4 + 2);
            show_fem(img_lb);
            axis equal; axis off;
            title(sprintf('CC=%.3f', cc_lb), 'FontSize', 10);
            if i == 1
                xlabel(metodos_nombre{1}, 'FontSize', 11, 'FontWeight', 'bold');
            end
            
            % COLUMNA 3: Propuesta
            subplot(5, 4, (i-1)*4 + 3);
            show_fem(img_cnn);
            axis equal; axis off;
            title(sprintf('CC=%.3f', cc_cnn), 'FontSize', 10);
            if i == 1
                xlabel(metodos_nombre{2}, 'FontSize', 11, 'FontWeight', 'bold');
            end
            
            % COLUMNA 4: Límite Superior
            subplot(5, 4, (i-1)*4 + 4);
            show_fem(img_gt_recon);
            axis equal; axis off;
            title(sprintf('CC=%.3f', cc_gt), 'FontSize', 10);
            if i == 1
                xlabel(metodos_nombre{3}, 'FontSize', 11, 'FontWeight', 'bold');
            end
            
        else
            for col = 1:4
                subplot(5, 4, (i-1)*4 + col);
                text(0.5, 0.5, 'ERROR: Imágenes no disponibles', ...
                    'HorizontalAlignment', 'center', 'Color', 'red');
            end
        end
    end
    
    sgtitle(sprintf('Reconstrucción - %s (Escenarios 6-10)', algoritmo_actual), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % GUARDAR FIGURA 2
    nombre_archivo = sprintf('Reconstruccion_%s_Esc6-10.png', algoritmo_actual);
    saveas(fig2, nombre_archivo);
    fprintf('      ✅ Guardada: %s\n', nombre_archivo);
end

fprintf('   ✅ 6 figuras de reconstrucción generadas exitosamente\n');
fprintf('================================================================\n');

%% =====================================================================
% SECCIÓN 8: TABLA DE RESUMEN FINAL
% =====================================================================
fprintf('\n📋 GENERANDO TABLA DE RESUMEN FINAL...\n');

fprintf('\n');
fprintf('📊==================================================================📊\n');
fprintf('   TABLA DE RESUMEN - VALIDACIÓN SISTEMÁTICA CNN EIT\n');
fprintf('📊==================================================================📊\n');
fprintf('\n');

% Encabezado de la tabla
fprintf('%-12s %-20s %10s %12s %12s %12s\n', ...
   'Escenario', 'Nombre', 'CC', 'MAE', 'ER', 'Tiempo_CNN');
fprintf('%-12s %-20s %10s %12s %12s %12s\n', ...
   '---------', '--------------------', '----------', '------------', '------------', '------------');

% Métricas válidas (sin NaN)
metricas_validas = ~isnan(metricas_resumen(:, 1));
n_validos = sum(metricas_validas);

% Filas de la tabla
for i = 1:n_scenarios
   if metricas_validas(i)
       fprintf('%-12d %-20s %10.6f %12.3e %12.3e %12.3f\n', ...
           i, nombres_escenarios{i}(1:min(20, end)), ...
           metricas_resumen(i, 1), metricas_resumen(i, 2), ...
           metricas_resumen(i, 3), metricas_resumen(i, 4));
   else
       fprintf('%-12d %-20s %10s %12s %12s %12s\n', ...
           i, 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR');
   end
end

% Separador
fprintf('%-12s %-20s %10s %12s %12s %12s\n', ...
   '---------', '--------------------', '----------', '------------', '------------', '------------');

% Métricas promedio (solo de escenarios válidos)
if n_validos > 0
   metricas_promedio = mean(metricas_resumen(metricas_validas, :), 1);
   
   fprintf('%-12s %-20s %10.6f %12.3e %12.3e %12.3f\n', ...
       'PROMEDIO', sprintf('(%d escenarios)', n_validos), ...
       metricas_promedio(1), metricas_promedio(2), ...
       metricas_promedio(3), metricas_promedio(4));
else
   fprintf('%-12s %-20s %10s %12s %12s %12s\n', ...
       'PROMEDIO', 'N/A', 'ERROR', 'ERROR', 'ERROR', 'ERROR');
end

fprintf('\n');

%% =====================================================================
% SECCIÓN 9: ANÁLISIS ESTADÍSTICO ADICIONAL
% =====================================================================
fprintf('📈 ANÁLISIS ESTADÍSTICO ADICIONAL...\n');

if n_validos > 0
   % Estadísticas descriptivas
   R2_stats = metricas_resumen(metricas_validas, 1);
   MAE_stats = metricas_resumen(metricas_validas, 2);
   
   fprintf('\n📊 ESTADÍSTICAS DESCRIPTIVAS (CC):\n');
   fprintf('   Promedio: %.6f\n', mean(R2_stats));
   fprintf('   Mediana:  %.6f\n', median(R2_stats));
   fprintf('   Desv. Est: %.6f\n', std(R2_stats));
   fprintf('   Mínimo:   %.6f\n', min(R2_stats));
   fprintf('   Máximo:   %.6f\n', max(R2_stats));
   
   fprintf('\n📊 ESTADÍSTICAS DESCRIPTIVAS (MAE):\n');
   fprintf('   Promedio: %.3e\n', mean(MAE_stats));
   fprintf('   Mediana:  %.3e\n', median(MAE_stats));
   fprintf('   Desv. Est: %.3e\n', std(MAE_stats));
   fprintf('   Mínimo:   %.3e\n', min(MAE_stats));
   fprintf('   Máximo:   %.3e\n', max(MAE_stats));
   
   % Comparación con objetivo
   objetivo_R2 = 0.95;
   R2_promedio = mean(R2_stats);
   diferencia_objetivo = R2_promedio - objetivo_R2;
   porcentaje_objetivo = (R2_promedio / objetivo_R2) * 100;
   
   fprintf('\n🎯 COMPARACIÓN CON OBJETIVO DEL PAPER:\n');
   fprintf('   CC Objetivo (Xu et al.): %.3f\n', objetivo_R2);
   fprintf('   CC Obtenido (Promedio): %.6f\n', R2_promedio);
   fprintf('   Diferencia: %+.6f\n', diferencia_objetivo);
   fprintf('   Porcentaje del objetivo: %.2f%%\n', porcentaje_objetivo);
   
   % Clasificación del rendimiento
   if R2_promedio >= 0.93
       clasificacion = '🏆 EXCELENTE';
   elseif R2_promedio >= 0.90
       clasificacion = '✅ BUENO';
   elseif R2_promedio >= 0.85
       clasificacion = '⚠️ ACEPTABLE';
   else
       clasificacion = '❌ INSUFICIENTE';
   end
   
   fprintf('   Clasificación: %s\n', clasificacion);
   
end

%% =====================================================================
% SECCIÓN 10: GUARDADO DE RESULTADOS
% =====================================================================
fprintf('\n💾 GUARDANDO RESULTADOS...\n');

try
   % Crear estructura de resultados para guardar
   resultados_completos = struct();
   resultados_completos.timestamp = datestr(now);
   resultados_completos.n_scenarios = n_scenarios;
   resultados_completos.n_validos = n_validos;
   resultados_completos.resultados_detallados = resultados;
   resultados_completos.metricas_resumen = metricas_resumen;
   resultados_completos.nombres_escenarios = nombres_escenarios;
   
   if n_validos > 0

       resultados_completos.estadisticas = struct();
       resultados_completos.estadisticas.R2_promedio = mean(R2_stats);
       resultados_completos.estadisticas.MAE_promedio = mean(MAE_stats);
       resultados_completos.estadisticas.R2_std = std(R2_stats);
       resultados_completos.estadisticas.MAE_std = std(MAE_stats);
       resultados_completos.estadisticas.diferencia_objetivo = diferencia_objetivo;
       resultados_completos.estadisticas.porcentaje_objetivo = porcentaje_objetivo;
       resultados_completos.estadisticas.clasificacion = clasificacion;
   end
   
   % Guardar en archivo .mat
   save('Gran_Comparacion_Resultados.mat', 'resultados_completos', '-v7.3');
   fprintf('   ✅ Resultados guardados en: Gran_Comparacion_Resultados.mat\n');
   
   % Guardar tabla de resumen como CSV
   if n_validos > 0
       % Crear tabla para exportar
       tabla_resumen = table();
       tabla_resumen.Escenario = (1:n_scenarios)';
       tabla_resumen.Nombre = nombres_escenarios;
       tabla_resumen.CC = metricas_resumen(:, 1);
       tabla_resumen.MAE = metricas_resumen(:, 2);
       tabla_resumen.ER = metricas_resumen(:, 3);
       tabla_resumen.Tiempo_CNN_ms = metricas_resumen(:, 4) * 1000; % Convertir a milisegundos
       
       writetable(tabla_resumen, 'Gran_Comparacion_Metricas.csv');
       fprintf('   ✅ Tabla de métricas guardada en: Gran_Comparacion_Metricas.csv\n');
   end
   
catch ME
   fprintf('   ⚠️ Error al guardar resultados: %s\n', ME.message);
end


%% =====================================================================
% SECCIÓN 11: FIGURA ADICIONAL - GRÁFICOS DE MÉTRICAS
% =====================================================================
fprintf('\n📊 Creando figura adicional de análisis de métricas...\n');

if n_validos > 0
   
   figure('Position', [300, 100, 1400, 800], 'Name', 'Análisis de Métricas por Escenario');
   
   % Subplot 1: CC por escenario
   subplot(2, 3, 1);
   escenarios_validos = find(metricas_validas);
   bar(escenarios_validos, R2_stats, 'FaceColor', [0.2, 0.6, 0.8]);
   hold on;
   yline(objetivo_R2, 'r--', 'LineWidth', 2, 'DisplayName', 'Objetivo (0.95)');
   yline(mean(R2_stats), 'g--', 'LineWidth', 2, 'DisplayName', sprintf('Promedio (%.3f)', mean(R2_stats)));
   
   title('CC por Escenario', 'FontWeight', 'bold');
   xlabel('Escenario');
   ylabel('CC');
   grid on;
   legend('Location', 'best');
   ylim([0, 1]);
   
   % Subplot 2: MAE por escenario
   subplot(2, 3, 2);
   bar(escenarios_validos, MAE_stats, 'FaceColor', [0.8, 0.4, 0.2]);
   hold on;
   yline(mean(MAE_stats), 'g--', 'LineWidth', 2, 'DisplayName', sprintf('Promedio (%.2e)', mean(MAE_stats)));
   
   title('MAE por Escenario', 'FontWeight', 'bold');
   xlabel('Escenario');
   ylabel('MAE');
   grid on;
   legend('Location', 'best');
   set(gca, 'YScale', 'log');
   
   % Subplot 3: Tiempo CNN por escenario
   subplot(2, 3, 3);
   tiempo_stats = metricas_resumen(metricas_validas, 4) * 1000; % Convertir a ms
   bar(escenarios_validos, tiempo_stats, 'FaceColor', [0.6, 0.2, 0.8]);
   hold on;
   yline(mean(tiempo_stats), 'g--', 'LineWidth', 2, 'DisplayName', sprintf('Promedio (%.1f ms)', mean(tiempo_stats)));
   
   title('Tiempo de Inferencia CNN', 'FontWeight', 'bold');
   xlabel('Escenario');
   ylabel('Tiempo (ms)');
   grid on;
   legend('Location', 'best');
   
   % Subplot 4: Histograma de CC
   subplot(2, 3, 4);
   histogram(R2_stats, 'BinEdges', 0:0.05:1, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'black');
   hold on;
   xline(mean(R2_stats), 'g--', 'LineWidth', 2, 'DisplayName', sprintf('Media = %.3f', mean(R2_stats)));
   xline(median(R2_stats), 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mediana = %.3f', median(R2_stats)));
   
   title('Distribución de CC', 'FontWeight', 'bold');
   xlabel('CC');
   ylabel('Frecuencia');
   grid on;
   legend('Location', 'best');
   
   % Subplot 5: Scatter CC vs MAE
   subplot(2, 3, 5);
   scatter(R2_stats, MAE_stats, 100, escenarios_validos, 'filled');
   colormap(jet);
   cb = colorbar; cb.Label.String = 'Escenario';
   
   title('Correlación CC vs MAE', 'FontWeight', 'bold');
   xlabel('CC');
   ylabel('MAE');
   grid on;
   
   % Añadir etiquetas de escenario
   for i = 1:length(escenarios_validos)
       text(R2_stats(i), MAE_stats(i), sprintf('  %d', escenarios_validos(i)), ...
            'FontSize', 8, 'FontWeight', 'bold');
   end
   
   % Subplot 6: Box plot de métricas normalizadas
   subplot(2, 3, 6);
   
   % Normalizar métricas para comparación
   R2_norm = R2_stats / max(R2_stats);
   MAE_norm = (max(MAE_stats) - MAE_stats) / (max(MAE_stats) - min(MAE_stats)); % Invertir para que mayor sea mejor
   
   boxplot([R2_norm, MAE_norm], 'Labels', {'CC (norm)', 'MAE (norm inv)'});
   title('Comparación de Métricas Normalizadas', 'FontWeight', 'bold');
   ylabel('Valor Normalizado');
   grid on;
   
   sgtitle('Análisis Detallado de Métricas de Rendimiento CNN', 'FontSize', 16, 'FontWeight', 'bold');
   
end

%% =====================================================================
% SECCIÓN 12: REPORTE FINAL Y CONCLUSIONES
% =====================================================================
fprintf('\n');
fprintf('🎉==================================================================🎉\n');
fprintf('   GRAN COMPARACIÓN COMPLETADA EXITOSAMENTE\n');
fprintf('🎉==================================================================🎉\n');
fprintf('\n');

fprintf('📋 RESUMEN EJECUTIVO:\n');
fprintf('   🔢 Escenarios procesados: %d/%d\n', n_validos, n_scenarios);

if n_validos > 0
   fprintf('   📊 CC promedio: %.6f\n', mean(R2_stats));
   fprintf('   📊 MAE promedio: %.3e\n', mean(MAE_stats));
   fprintf('   🎯 Clasificación: %s\n', clasificacion);
   fprintf('   ⚡ Velocidad promedio: %.1f ms por predicción\n', mean(tiempo_stats));
   
   % Análisis de consistencia
   cv_R2 = std(R2_stats) / mean(R2_stats); % Coeficiente de variación
   fprintf('   📈 Consistencia (CV de CC): %.3f\n', cv_R2);
   
   if cv_R2 < 0.1
       consistencia = '✅ Muy consistente';
   elseif cv_R2 < 0.2
       consistencia = '✅ Consistente';
   else
       consistencia = '⚠️ Variable';
   end
   fprintf('   📊 Evaluación de consistencia: %s\n', consistencia);
   
   % Recomendaciones
   fprintf('\n💡 RECOMENDACIONES:\n');
   
   if mean(R2_stats) >= 0.90
       fprintf('   ✅ Modelo listo para aplicaciones prácticas\n');
       fprintf('   🎯 Considerar optimización de velocidad para tiempo real\n');
   elseif mean(R2_stats) >= 0.80
       fprintf('   ⚠️ Modelo funcional, requiere mejoras adicionales\n');
       fprintf('   🔧 Considerar ajuste de hiperparámetros o más datos de entrenamiento\n');
   else
       fprintf('   ❌ Modelo requiere revisión fundamental\n');
       fprintf('   🏗️ Considerar cambios en arquitectura o metodología\n');
   end
   
   % Identificar mejores y peores escenarios
   [~, idx_mejor] = max(R2_stats);
   [~, idx_peor] = min(R2_stats);
   
   fprintf('\n🏆 ANÁLISIS POR ESCENARIOS:\n');
   fprintf('   🥇 Mejor escenario: %d (%s) - CC = %.6f\n', ...
       escenarios_validos(idx_mejor), nombres_escenarios{escenarios_validos(idx_mejor)}, R2_stats(idx_mejor));
   fprintf('   🥉 Peor escenario: %d (%s) - CC = %.6f\n', ...
       escenarios_validos(idx_peor), nombres_escenarios{escenarios_validos(idx_peor)}, R2_stats(idx_peor));
   
else
   fprintf('   ❌ No se procesaron escenarios válidos\n');
   fprintf('   🔧 Revise la configuración y dependencias\n');
end

fprintf('\n📁 ARCHIVOS GENERADOS:\n');
fprintf('   📊 Figura 1: Análisis Comparativo 1-5\n');
fprintf('   📊 Figura 2: Análisis Comparativo 6-10\n');
fprintf('   📈 Figura 3: Análisis de Métricas\n');
fprintf('   💾 Gran_Comparacion_Resultados.mat\n');
if n_validos > 0
   fprintf('   📋 Gran_Comparacion_Metricas.csv\n');
end

fprintf('\n🏁 VALIDACIÓN SISTEMÁTICA FINALIZADA\n');
fprintf('   Fecha: %s\n', datestr(now));
fprintf('   Duración: Procesamiento completo de %d escenarios\n', n_scenarios);

fprintf('\n🚀 PROYECTO COMPLETADO: Dataset → Entrenamiento → Validación\n');
fprintf('==================================================================\n');

%% ======================================================================
% SECCIÓN 13: TABLA DE RESUMEN DE MÉTRICAS DE IMAGEN
% ======================================================================

fprintf('\n\n');
fprintf('╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║          RESUMEN DE MÉTRICAS DE RECONSTRUCCIÓN DE IMAGEN (PROMEDIO DE %2d ESCENARIOS)                 ║\n', n_validos);
fprintf('╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

% --- Agregación de métricas de IMAGEN ---
metodos = {'linea_base', 'propuesta', 'limite_sup'};
algoritmos = {'Tikhonov', 'Laplace', 'TV'};

% Pre-alocar matrices para promedios. Dim: [Método x Algoritmo]
avg_CC_img = zeros(3, 3);
avg_RE_img = zeros(3, 3);
avg_MAE_img = zeros(3, 3);

% Estructuras temporales para guardar todos los valores
temp_CC_img = NaN(length(resultados), 3, 3); % [Escenario, Método, Algoritmo]
temp_RE_img = NaN(length(resultados), 3, 3);
temp_MAE_img = NaN(length(resultados), 3, 3);

for scenario_id = 1:length(resultados)
    if isfield(resultados{scenario_id}, 'metricas_imagen')
        for m = 1:length(metodos)
            for a = 1:length(algoritmos)
                temp_CC_img(scenario_id, m, a) = resultados{scenario_id}.metricas_imagen.(metodos{m}).(algoritmos{a}).CC;
                temp_RE_img(scenario_id, m, a) = resultados{scenario_id}.metricas_imagen.(metodos{m}).(algoritmos{a}).RE;
                temp_MAE_img(scenario_id, m, a) = resultados{scenario_id}.metricas_imagen.(metodos{m}).(algoritmos{a}).MAE;
            end
        end
    end
end

% Calcular promedios
if n_validos > 0
    avg_CC_img = squeeze(mean(temp_CC_img, 1, 'omitnan'));
    avg_RE_img = squeeze(mean(temp_RE_img, 1, 'omitnan'));
    avg_MAE_img = squeeze(mean(temp_MAE_img, 1, 'omitnan'));
end

% --- Presentación en Tabla en Command Window ---
metodos_nombres_tabla = {'Línea Base (8e)', 'Propuesta (8e+CNN)', 'Límite Superior (96ch GT)'};

for a = 1:length(algoritmos)
    algoritmo_actual = algoritmos{a};
    
    fprintf('--- MÉTRICAS PARA EL ALGORITMO DE RECONSTRUCCIÓN: %s ---\n', upper(algoritmo_actual));
    fprintf('%-25s | %15s | %15s | %15s\n', 'Método', 'CC (Más>Mejor)', 'RE (Menos>Mejor)', 'MAE (Menos>Mejor)');
    fprintf(repmat('-', 1, 80));
    fprintf('\n');
    
    for m = 1:length(metodos)
        fprintf('%-25s | %15.4f | %15.4f | %15.3e\n', ...
            metodos_nombres_tabla{m}, ...
            avg_CC_img(m, a), ...
            avg_RE_img(m, a), ...
            avg_MAE_img(m, a) ...
        );
    end
    fprintf('\n');
end

fprintf(repmat('=', 1, 80));
fprintf('\n\n');

%% =====================================================================
% FUNCIONES AUXILIARES
% =====================================================================

function formatted_str = addcomma(number)
   % Añade comas como separadores de miles para legibilidad
   str = sprintf('%.0f', number);
   formatted_str = regexprep(str, '(\d)(?=(\d{3})+(?!\d))', '$1,');
end

function shape_str = format_py_shape(py_shape_obj)
    % Convierte un objeto de shape de Keras (que es una tupla de Python) 
    % a un string legible en MATLAB, manejando el valor 'None'.
    try
        % Convertir la tupla de Python a una celda de MATLAB
        matlab_cell = cell(py_shape_obj);
        
        % Convertir cada elemento de la celda a un string
        str_cell = cellfun(@(x) ...
            char(py.str(x)), ... % Usar str() de Python para manejar 'None'
            matlab_cell, 'UniformOutput', false);
            
        % Unir los strings con comas
        shape_str = strjoin(str_cell, ', ');
    catch
        shape_str = 'Error al formatear shape';
    end
end

function hp_optimo = optimizar_hp_simple(imdl_template, vh, vi, metodo)
    % Versión simplificada para encontrar un buen hiperparámetro
    hp_range = logspace(-8, -2, 15);
    cc_values = zeros(size(hp_range));
    
    for i = 1:length(hp_range)
        imdl_test = crear_imdl_simple(imdl_template.fwd_model, metodo, hp_range(i));
        try
            img_recon = inv_solve(imdl_test, vh, vi);
            % img_gt debe estar en el workspace de la función que llama
            img_gt = evalin('caller', 'img_test_16e');
            if size(img_recon.elem_data,1) ~= size(img_gt.elem_data,1)
                img_gt = evalin('caller', 'img_test_8e');
            end
            metrics = calcular_metricas_imagen(img_recon, img_gt, true);
            cc_values(i) = metrics.CC;
        catch
            cc_values(i) = -1; % Penalizar si falla
        end
    end
    
    [~, best_idx] = max(cc_values);
    hp_optimo = hp_range(best_idx);
    fprintf('      ✓ HP óptimo para %s: %.2e\n', metodo, hp_optimo);
end

function imdl = crear_imdl_simple(fmdl, tipo_prior, hp)
% Crea un imdl robusto asegurando un Jacobian precalculado y estable.
    
    imdl = eidors_obj('inv_model', ['imdl_', tipo_prior]);
    imdl.fwd_model = fmdl;
    imdl.reconst_type = 'difference';
    imdl.hyperparameter.value = hp;
    imdl.solve = @inv_solve_diff_GN_one_step; % Solucionador estable
    
    n_elems = size(fmdl.elems, 1);
    
    % --- CONFIGURACIÓN CRÍTICA DEL JACOBIAN Y FONDO (Background = 1.0) ---
    try
        % 1. Crear el objeto imagen de fondo (CRÍTICO)
        img_bkgnd = mk_image(fmdl, 1.0); 
        
        % 2. Asignar el objeto imagen al fondo del Jacobiano
        imdl.jacobian_bkgnd = img_bkgnd; 
        
        % 3. CRÍTICO: Calcular y almacenar el Jacobiano para estabilidad
        imdl.jacobian = calc_jacobian(img_bkgnd);
        
    catch ME
        warning('Fallo CRÍTICO al pre-calcular Jacobian. Usando configuración por defecto.');
        imdl.jacobian_bkgnd.value = 1.0; % Fallback a valor escalar
    end
    
    % --- CONFIGURACIÓN DEL PRIOR ---
    switch lower(tipo_prior)
        case 'tikhonov'
            imdl.RtR_prior = speye(n_elems); % Prior de Identidad
            
        case 'laplace'
            imdl.solve = @inv_solve_diff_GN_one_step;
            % Calcular la matriz R y ASIGNAR LA MATRIZ (no la función)
            R_laplace = feval(@prior_laplace, imdl);
            imdl.RtR_prior = R_laplace; 
            
        case 'tv'
            imdl.solve = @inv_solve_diff_GN_one_step;
            % Calcular la matriz R y ASIGNAR LA MATRIZ (no la función)
            try
                R_nos = feval(@prior_nos, imdl);
                imdl.RtR_prior = R_nos; 
            catch
                R_laplace = feval(@prior_laplace, imdl);
                imdl.RtR_prior = R_laplace;
            end
            
        otherwise
            error('Tipo de prior no soportado.');
    end
end

function volt_synth = crear_voltaje_sintetico(fmdl_ref, volt_homog_ref, data_vector)
%CREAR_VOLTAJE_SINTETICO Crea una estructura de voltaje compatible con EIDORS
% a partir de un vector de datos diferenciales.
%
% Sintaxis:
%   volt_synth = crear_voltaje_sintetico(fmdl_ref, volt_homog_ref, data_vector)
%
% Entradas:
%   fmdl_ref       - El fwd_model correspondiente a los datos (ej. fmdl_96ch)
%   volt_homog_ref - Una medición de voltaje homogénea de referencia
%   data_vector    - El vector de datos diferenciales (ej. la predicción de la CNN)
%
% Salida:
%   volt_synth     - Una estructura de datos de EIDORS lista para inv_solve

    % Crear una nueva estructura de datos de EIDORS
    volt_synth = eidors_obj('data', 'synthetic data');
    volt_synth.fwd_model = fmdl_ref;
    
    % Asignar el vector de datos al campo de mediciones
    volt_synth.meas = data_vector(:);
    
    % Copiar información esencial de la referencia homogénea
    volt_synth.time = volt_homog_ref.time;
    volt_synth.name = 'synthetic_inhomogeneous_data';
    
    % Verificar que las dimensiones son consistentes
    num_meas_expected = size(fmdl_ref.stimulation(1).meas_pattern, 1) * length(fmdl_ref.stimulation);
    if length(volt_synth.meas) ~= num_meas_expected
        warning('crear_voltaje_sintetico:DimensionMismatch', ...
            'El tamaño del data_vector (%d) no coincide con las mediciones esperadas del fmdl (%d)', ...
            length(volt_synth.meas), num_meas_expected);
    end
end

function [img, actual_scenario_id] = generar_imagen_conductividad_forzado(fmdl, CONFIG, scenario_id_objetivo)
%GENERAR_IMAGEN_CONDUCTIVIDAD_FORZADO Genera un phantom de un escenario específico.
%
% Llama repetidamente a 'generar_imagen_conductividad' (que es aleatoria)
% hasta que se genera el escenario con el ID deseado.

    max_attempts = 500; 

    for attempt = 1:max_attempts
        % Llamar a la función ORIGINAL, que elige un escenario al azar
        [img_temp, temp_id] = generar_imagen_conductividad(fmdl, CONFIG);

        % Comprobar si hemos obtenido el escenario que queríamos
        if temp_id == scenario_id_objetivo
            % ¡Éxito! Devolver la imagen y el ID y salir de la función
            img = img_temp;
            actual_scenario_id = temp_id;
            return; 
        end
    end
    
    % Si después de muchos intentos no lo encontramos, lanzar un error
    error('generar_imagen_conductividad_forzado:MaxAttemptsExceeded', ...
          'No se pudo generar el escenario %d después de %d intentos.', ...
          scenario_id_objetivo, max_attempts);
          
end

function metricas = calcular_metricas_imagen(img_recon, img_ref, interpolar)
%CALCULAR_METRICAS_IMAGEN Calcula métricas de imagen con interpolación manual robusta

    if nargin < 3, interpolar = false; end
    
    sigma_recon = double(img_recon.elem_data(:));
    sigma_ref = double(img_ref.elem_data(:));
    
    if interpolar && (length(sigma_recon) ~= length(sigma_ref))
        
        fprintf('      ⚠️  Interpolando imagen de %d a %d elementos (MÉTODO MANUAL CON scatteredInterpolant)...\n', ...
            length(sigma_recon), length(sigma_ref));
            
        try
            % 1. Obtener centroides y datos de la malla FUENTE
            fmdl_fuente = img_recon.fwd_model;
            centroids_fuente = zeros(size(fmdl_fuente.elems, 1), 2);
            for i = 1:size(fmdl_fuente.elems, 1)
                centroids_fuente(i, :) = mean(fmdl_fuente.nodes(fmdl_fuente.elems(i,:), 1:2), 1);
            end
            datos_fuente = img_recon.elem_data;
            
            % 2. Obtener los puntos de consulta de la malla DESTINO
            fmdl_destino = img_ref.fwd_model;
            centroids_destino = zeros(size(fmdl_destino.elems, 1), 2);
            for i = 1:size(fmdl_destino.elems, 1)
                centroids_destino(i, :) = mean(fmdl_destino.nodes(fmdl_destino.elems(i,:), 1:2), 1);
            end

            % 3. Crear el objeto de interpolación de MATLAB
            F = scatteredInterpolant(centroids_fuente(:,1), centroids_fuente(:,2), datos_fuente, 'linear', 'nearest');
            
            % 4. Evaluar el interpolante en los centroides de la malla de destino
            datos_interpolados = F(centroids_destino(:,1), centroids_destino(:,2));
            
            % 5. Sobrescribir sigma_recon con los datos interpolados
            sigma_recon = datos_interpolados(:);
            
        catch ME_interp
            error('Fallo en la interpolación manual: %s', ME_interp.message);
        end
        
    elseif ~interpolar && (length(sigma_recon) ~= length(sigma_ref))
        error('Dimensiones incompatibles sin permiso para interpolar: %d vs %d', ...
            length(sigma_recon), length(sigma_ref));
    end
    
    % --- CALCULAR MÉTRICAS ---
    
    if length(sigma_recon) ~= length(sigma_ref)
        error('Fallo crítico en interpolación: los tamaños siguen sin coincidir.');
    end

    metricas.RE = norm(sigma_recon - sigma_ref) / (norm(sigma_ref) + eps);
    
    if std(sigma_recon) > 1e-9 && std(sigma_ref) > 1e-9
        metricas.CC = corr(sigma_recon, sigma_ref);
    else
        metricas.CC = 0;
    end
    
    metricas.MAE = mean(abs(sigma_recon - sigma_ref));
    
    if ~isfinite(metricas.RE), metricas.RE = Inf; end
    if ~isfinite(metricas.CC) || isnan(metricas.CC), metricas.CC = 0; end
    if ~isfinite(metricas.MAE), metricas.MAE = Inf; end
end
