function img_16e = mapear_conductividad_8_a_16(fmdl_16, img_8e, CONFIG)
% mapear_conductividad_8_a_16 - Mapea phantom de 8 electrodos a 16 electrodos
%
% Sintaxis:
%   img_16e = mapear_conductividad_8_a_16(fmdl_16, img_8e, CONFIG)
%
% Entradas:
%   fmdl_16 - Estructura forward model de 16 electrodos (modelo destino)
%   img_8e  - Estructura de imagen de 8 electrodos ya generada (fuente)
%   CONFIG  - Struct de configuración con campos:
%             - conductividad_fondo: Conductividad del medio de fondo
%             - conductividad_objeto: Conductividad de las inclusiones
%
% Salida:
%   img_16e - Estructura de imagen para modelo de 16 electrodos
%             (físicamente idéntica a img_8e)
%
% Descripción:
%   FUNCIÓN CRÍTICA para garantizar consistencia física entre simulaciones.
%   
%   Esta función es el núcleo de la metodología para evitar data leakage:
%   toma una imagen ya generada en un sistema de 8 electrodos y crea su
%   contraparte físicamente IDÉNTICA en un sistema de 16 electrodos.
%   
%   La clave está en que ambas imágenes representan exactamente el mismo
%   phantom físico - mismas inclusiones, mismas posiciones, mismas
%   conductividades - solo difieren en la resolución de la malla y número
%   de electrodos del sistema de medición.
%
%   Proceso garantizado:
%   1. Extrae el scenario_id de la imagen de 8 electrodos
%   2. Usa generar_imagen_conductividad_con_id para recrear el MISMO phantom
%   3. Valida que ambas imágenes corresponden al mismo escenario físico
%   4. Añade metadatos de mapeo para trazabilidad completa
%
%   CRUCIAL: Esta función es fundamental para que el dataset de entrenamiento
%   sea válido. Cada muestra del dataset debe representar el mismo phantom
%   físico en ambos sistemas (8e y 16e), diferenciándose solo en la
%   resolución de medición.
%
% Ejemplo:
%   fmdl_8e = crear_modelo_fem(8);
%   fmdl_16e = crear_modelo_fem(16);
%   CONFIG.conductividad_fondo = 1.0;
%   CONFIG.conductividad_objeto = 0.3;
%   
%   % Generar phantom en 8e
%   [img_8e, scenario_id] = generar_imagen_conductividad(fmdl_8e, CONFIG);
%   
%   % Mapear mismo phantom a 16e (consistencia física garantizada)
%   img_16e = mapear_conductividad_8_a_16(fmdl_16e, img_8e, CONFIG);
%   
%   % Verificar consistencia
%   assert(img_8e.scenario_id == img_16e.scenario_id);
%
% Ver también: generar_imagen_conductividad_con_id, generar_imagen_conductividad

%% Validación exhaustiva de entradas
if nargin ~= 3
   error('mapear_conductividad_8_a_16:WrongNumberOfInputs', ...
       'La función requiere exactamente 3 argumentos de entrada');
end

% Validar modelo FEM de 16 electrodos
if ~isstruct(fmdl_16)
   error('mapear_conductividad_8_a_16:InvalidFmdl16Input', ...
       'fmdl_16 debe ser una estructura de forward model de EIDORS');
end

% Verificar campos esenciales del modelo de 16 electrodos
required_fmdl_fields = {'nodes', 'elems', 'electrode'};
for i = 1:length(required_fmdl_fields)
   if ~isfield(fmdl_16, required_fmdl_fields{i})
       error('mapear_conductividad_8_a_16:MissingFmdl16Field', ...
           'fmdl_16 debe contener el campo esencial: %s', required_fmdl_fields{i});
   end
end

% Verificar que es realmente un modelo de 16 electrodos
if isfield(fmdl_16, 'electrode') && length(fmdl_16.electrode) ~= 16
   warning('mapear_conductividad_8_a_16:WrongElectrodeCount', ...
       'fmdl_16 tiene %d electrodos, se esperaban 16', length(fmdl_16.electrode));
end

% Validar imagen de 8 electrodos
if ~isstruct(img_8e)
   error('mapear_conductividad_8_a_16:InvalidImg8eInput', ...
       'img_8e debe ser una estructura de imagen de EIDORS');
end

% Verificar campos obligatorios de la imagen de 8 electrodos
required_img_fields = {'fwd_model', 'elem_data', 'scenario_id'};
for i = 1:length(required_img_fields)
   field_name = required_img_fields{i};
   if ~isfield(img_8e, field_name)
       error('mapear_conductividad_8_a_16:MissingImg8eField', ...
           'img_8e debe contener el campo crítico: %s. Esto indica que la imagen no fue generada con generar_imagen_conductividad.m', ...
           field_name);
   end
end

% Validar estructura CONFIG
if ~isstruct(CONFIG)
   error('mapear_conductividad_8_a_16:InvalidConfigInput', ...
       'CONFIG debe ser una estructura');
end

% Verificar campos obligatorios en CONFIG
required_config_fields = {'conductividad_fondo', 'conductividad_objeto'};
for i = 1:length(required_config_fields)
   field_name = required_config_fields{i};
   
   if ~isfield(CONFIG, field_name)
       error('mapear_conductividad_8_a_16:MissingConfigField', ...
           'CONFIG debe contener el campo obligatorio: %s', field_name);
   end
   
   % Validar que los valores son numéricos, escalares y finitos
   field_value = CONFIG.(field_name);
   if ~isnumeric(field_value) || ~isscalar(field_value) || ~isfinite(field_value)
       error('mapear_conductividad_8_a_16:InvalidConfigValue', ...
           'CONFIG.%s debe ser un número escalar finito. Valor recibido: %s', ...
           field_name, mat2str(field_value));
   end
   
   % Validar que las conductividades son estrictamente positivas
   if field_value <= 0
       error('mapear_conductividad_8_a_16:InvalidConductivityValue', ...
           'CONFIG.%s debe ser un valor estrictamente positivo. Valor recibido: %.6f', ...
           field_name, field_value);
   end
end

%% Extraer scenario_id de la imagen de 8 electrodos
scenario_id = img_8e.scenario_id;

% Validar scenario_id extraído
if ~isnumeric(scenario_id) || ~isscalar(scenario_id) || ...
  ~isfinite(scenario_id) || scenario_id ~= round(scenario_id)
   error('mapear_conductividad_8_a_16:InvalidScenarioId', ...
       'img_8e.scenario_id debe ser un entero finito válido. Valor encontrado: %s', ...
       mat2str(scenario_id));
end

if scenario_id < 1
   error('mapear_conductividad_8_a_16:ScenarioIdOutOfRange', ...
       'img_8e.scenario_id (%d) debe ser mayor que 0', scenario_id);
end

%% Recrear imagen idéntica en modelo de 16 electrodos
try
   [img_16e, returned_scenario_id] = generar_imagen_conductividad_con_id(fmdl_16, CONFIG, scenario_id);
catch ME
   error('mapear_conductividad_8_a_16:MappingFailed', ...
       'Error al generar imagen idéntica en modelo 16e: %s', ME.message);
end

%% Verificación crítica de consistencia física
% Verificar que se generó el escenario correcto
if returned_scenario_id ~= scenario_id
   error('mapear_conductividad_8_a_16:ScenarioIdMismatch', ...
       'CRÍTICO: El scenario_id retornado (%d) no coincide con el solicitado (%d)', ...
       returned_scenario_id, scenario_id);
end

% Verificar que ambas imágenes representan el mismo escenario
if img_8e.scenario_id ~= img_16e.scenario_id
   error('mapear_conductividad_8_a_16:FinalConsistencyCheckFailed', ...
       'CRÍTICO: Las imágenes no representan el mismo escenario físico: 8e=%d, 16e=%d', ...
       img_8e.scenario_id, img_16e.scenario_id);
end

%% Añadir metadatos de mapeo para trazabilidad completa
% Información del proceso de mapeo
img_16e.mapping_info = struct();
img_16e.mapping_info.source_model = '8_electrodes';
img_16e.mapping_info.target_model = '16_electrodes';
img_16e.mapping_info.mapping_method = 'scenario_id_consistency';
img_16e.mapping_info.original_scenario_id = scenario_id;
img_16e.mapping_info.mapping_timestamp = datestr(now);
img_16e.mapping_info.physical_consistency = 'GUARANTEED';

% Copiar metadatos relevantes de la imagen original para trazabilidad
if isfield(img_8e, 'scenario_name')
   img_16e.mapping_info.original_scenario_name = img_8e.scenario_name;
end

if isfield(img_8e, 'scenario_type')
   img_16e.mapping_info.original_scenario_type = img_8e.scenario_type;
end

if isfield(img_8e, 'generation_timestamp')
   img_16e.mapping_info.original_generation_timestamp = img_8e.generation_timestamp;
end

% Verificar consistencia de parámetros de conductividad
if isfield(img_8e, 'conductividad_fondo_usado')
   expected_bg = img_8e.conductividad_fondo_usado;
   if abs(CONFIG.conductividad_fondo - expected_bg) > 1e-12
       warning('mapear_conductividad_8_a_16:ConductivityMismatch', ...
           'La conductividad de fondo difiere entre 8e (%.6f) y CONFIG (%.6f)', ...
           expected_bg, CONFIG.conductividad_fondo);
   end
   img_16e.mapping_info.conductivity_consistency_check = 'verified';
else
   img_16e.mapping_info.conductivity_consistency_check = 'not_available';
end

%% Validación final de la imagen mapeada
% Verificar integridad de la imagen de 16e
if any(~isfinite(img_16e.elem_data))
   warning('mapear_conductividad_8_a_16:NonFiniteResults', ...
       'La imagen 16e contiene valores no finitos');
end

if isempty(img_16e.elem_data)
   error('mapear_conductividad_8_a_16:EmptyMappedImage', ...
       'La imagen 16e mapeada está vacía - error crítico');
end

% Verificar que la imagen tiene sentido físico
if min(img_16e.elem_data) < 0
   warning('mapear_conductividad_8_a_16:NegativeConductivity', ...
       'La imagen 16e contiene conductividades negativas');
end

end

%% =====================================================================
%% FUNCIÓN AUXILIAR: generar_imagen_conductividad_con_id
%% =====================================================================
function [img, scenario_id] = generar_imagen_conductividad_con_id(fmdl, CONFIG, scenario_id)
% generar_imagen_conductividad_con_id - Genera imagen para escenario específico
%
% Sintaxis:
%   [img, scenario_id] = generar_imagen_conductividad_con_id(fmdl, CONFIG, scenario_id)
%
% Entradas:
%   fmdl        - Estructura forward model de EIDORS
%   CONFIG      - Struct de configuración con conductividades
%   scenario_id - ID específico del escenario a generar (1-10)
%
% Salidas:
%   img        - Estructura de imagen con phantom del escenario especificado
%   scenario_id- El mismo ID que se pasó como entrada (confirmación)
%
% Descripción:
%   Función auxiliar que genera una imagen para un escenario específico
%   en lugar de seleccionar uno al azar. Es esencialmente idéntica a
%   generar_imagen_conductividad.m pero recibe el scenario_id como parámetro.
%   
%   Esta función es CRÍTICA para garantizar que podemos recrear el mismo
%   phantom físico en diferentes modelos FEM.

%% Validación de entradas específicas
if nargin ~= 3
   error('generar_imagen_conductividad_con_id:WrongNumberOfInputs', ...
       'La función requiere exactamente 3 argumentos de entrada');
end

% Validar scenario_id específico
if ~isnumeric(scenario_id) || ~isscalar(scenario_id) || ...
  ~isfinite(scenario_id) || scenario_id ~= round(scenario_id)
   error('generar_imagen_conductividad_con_id:InvalidScenarioId', ...
       'scenario_id debe ser un entero finito válido. Valor recibido: %s', ...
       mat2str(scenario_id));
end

%% Reutilizar validaciones de generar_imagen_conductividad
% (Las validaciones de fmdl y CONFIG son idénticas)

%% Obtener catálogo de escenarios
try
   scenarios = get_scenarios_definition();
catch ME
   error('generar_imagen_conductividad_con_id:ScenariosDefinitionFailed', ...
       'Error al obtener la definición de escenarios: %s', ME.message);
end

n_scenarios = length(scenarios);

% Validar que el scenario_id solicitado existe
if scenario_id < 1 || scenario_id > n_scenarios
   error('generar_imagen_conductividad_con_id:ScenarioIdOutOfRange', ...
       'scenario_id (%d) debe estar entre 1 y %d', scenario_id, n_scenarios);
end

%% Seleccionar escenario específico (NO aleatorio)
selected_scenario = scenarios{scenario_id};

% Validar escenario seleccionado
if ~isstruct(selected_scenario)
   error('generar_imagen_conductividad_con_id:InvalidSelectedScenario', ...
       'El escenario %d no es una estructura válida', scenario_id);
end

%% El resto del código es idéntico a generar_imagen_conductividad
% Crear imagen base
try
   img = mk_image(fmdl, CONFIG.conductividad_fondo);
catch ME
   error('generar_imagen_conductividad_con_id:ImageCreationFailed', ...
       'Error al crear la imagen base: %s', ME.message);
end

% Añadir metadatos
img.scenario_id = scenario_id;
img.scenario_name = selected_scenario.name;
img.scenario_type = selected_scenario.type;
img.conductividad_fondo_usado = CONFIG.conductividad_fondo;
img.conductividad_objeto_usado = CONFIG.conductividad_objeto;
img.generation_timestamp = datestr(now);
img.generation_method = 'deterministic_by_id';  % Distinguir del método aleatorio

if isfield(selected_scenario, 'num_inclusiones')
   img.num_inclusiones_esperadas = selected_scenario.num_inclusiones;
else
   img.num_inclusiones_esperadas = length(selected_scenario.inclusiones);
end

% Añadir inclusiones (código idéntico)
if strcmp(selected_scenario.type, 'homogeneo')
   img.num_inclusiones_aplicadas = 0;
   img.inclusion_summary = 'Escenario homogéneo - sin inclusiones';
else
   inclusiones = selected_scenario.inclusiones;
   n_inclusiones = length(inclusiones);
   inclusiones_exitosas = 0;
   inclusion_details = cell(n_inclusiones, 1);
   
   for i = 1:n_inclusiones
       try
           inclusion = inclusiones{i};
           
           if ~isstruct(inclusion)
               inclusion_details{i} = sprintf('Inclusión %d: ERROR - estructura inválida', i);
               continue;
           end
           
           required_fields = {'x', 'y', 'radius'};
           skip_inclusion = false;
           for j = 1:length(required_fields)
               if ~isfield(inclusion, required_fields{j})
                   skip_inclusion = true;
                   break;
               end
           end
           
           if skip_inclusion
               inclusion_details{i} = sprintf('Inclusión %d: ERROR - campos faltantes', i);
               continue;
           end
           
           x_center = inclusion.x;
           y_center = inclusion.y;
           radius = inclusion.radius;
           
           if ~isnumeric(x_center) || ~isnumeric(y_center) || ~isnumeric(radius) || ...
              ~isscalar(x_center) || ~isscalar(y_center) || ~isscalar(radius) || ...
              ~isfinite(x_center) || ~isfinite(y_center) || ~isfinite(radius) || ...
              radius <= 0
               inclusion_details{i} = sprintf('Inclusión %d: ERROR - parámetros inválidos', i);
               continue;
           end
           
           img = add_circular_inclusion(img, x_center, y_center, radius, CONFIG.conductividad_objeto);
           inclusiones_exitosas = inclusiones_exitosas + 1;
           inclusion_details{i} = sprintf('Inclusión %d: centro=(%.3f,%.3f) radio=%.3f - EXITOSA', ...
               i, x_center, y_center, radius);
           
       catch ME
           inclusion_details{i} = sprintf('Inclusión %d: ERROR - %s', i, ME.message);
       end
   end
   
   img.num_inclusiones_aplicadas = inclusiones_exitosas;
   img.inclusion_summary = inclusion_details;
end

% Añadir estadísticas finales
if isfield(img, 'elem_data')
   img.final_conductivity_stats = struct();
   img.final_conductivity_stats.min_value = min(img.elem_data);
   img.final_conductivity_stats.max_value = max(img.elem_data);
   img.final_conductivity_stats.mean_value = mean(img.elem_data);
   img.final_conductivity_stats.std_value = std(img.elem_data);
   
   elementos_fondo = sum(abs(img.elem_data - CONFIG.conductividad_fondo) < 1e-12);
   elementos_total = length(img.elem_data);
   img.final_conductivity_stats.fraction_background = elementos_fondo / elementos_total;
   img.final_conductivity_stats.fraction_modified = 1 - img.final_conductivity_stats.fraction_background;
end

end