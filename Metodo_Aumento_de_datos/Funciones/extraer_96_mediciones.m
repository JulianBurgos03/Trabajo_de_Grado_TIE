function mediciones_96 = extraer_96_mediciones(volt_data_16e)
% extraer_96_mediciones - Extrae 96 mediciones específicas del sistema de 16 electrodos
%
% Sintaxis:
%   mediciones_96 = extraer_96_mediciones(volt_data_16e)
%
% Entrada:
%   volt_data_16e - Estructura de datos retornada por fwd_solve de EIDORS
%                   para una simulación de 16 electrodos (208 mediciones esperadas)
%
% Salida:
%   mediciones_96 - Vector columna de 96x1 con mediciones extraídas según
%                   metodología de canales virtuales
%
% Descripción:
%   FUNCIÓN CRÍTICA para la validez del dataset de entrenamiento.
%   
%   Esta función implementa la metodología específica del paper Xu et al. (2022)
%   "A Virtual Channel Based Data Augmentation Method for EIT" para extraer
%   mediciones que simulan un sistema de 8 excitaciones virtuales a partir
%   de un sistema real de 16 electrodos.
%
%   IMPORTANCIA CRÍTICA - Prevención de Data Leakage:
%   El objetivo NO es usar todas las 208 mediciones disponibles del sistema
%   de 16 electrodos, ya que esto introduciría información que no está
%   disponible en el sistema de 8 electrodos, causando data leakage y
%   invalidando el entrenamiento de la CNN.
%
%   METODOLOGÍA ESPECÍFICA (Xu et al., 2022):
%   
%   Sistema de 16 electrodos produce:
%   - 16 excitaciones (una por cada par de electrodos adyacentes)
%   - 13 mediciones por excitación (pares adyacentes no usados para estimulación)
%   - Total: 16 × 13 = 208 mediciones
%
%   Extracción de canales virtuales:
%   - Seleccionar excitaciones ALTERNADAS: 1, 3, 5, 7, 9, 11, 13, 15
%   - De cada excitación seleccionada: tomar las PRIMERAS 12 mediciones
%   - Resultado: 8 excitaciones × 12 mediciones = 96 mediciones
%
%   JUSTIFICACIÓN DE LA METODOLOGÍA:
%   1. Excitaciones alternadas (1,3,5,7,9,11,13,15) simulan un patrón
%      de excitación realista que podría implementarse físicamente
%   2. Tomar 12 de 13 mediciones mantiene alta resolución sin usar
%      información "imposible" para un sistema de 8 electrodos  
%   3. El patrón sistemático (no aleatorio) garantiza reproducibilidad
%   4. La selección evita usar información privilegiada del sistema de 16e
%
%   ESTRUCTURA ESPERADA DE DATOS 16e:
%   Excitación 1:  mediciones 1-13    →  tomar mediciones 1-12
%   Excitación 2:  mediciones 14-26   →  SALTAR (no alternada)
%   Excitación 3:  mediciones 27-39   →  tomar mediciones 27-38
%   Excitación 4:  mediciones 40-52   →  SALTAR (no alternada)
%   ...
%   Excitación 15: mediciones 183-195 →  tomar mediciones 183-194
%   Excitación 16: mediciones 196-208 →  SALTAR (no alternada)
%
%   Esta metodología es FUNDAMENTAL para:
%   - Mantener consistencia física entre sistemas 8e y 16e
%   - Evitar data leakage en el entrenamiento de la CNN
%   - Replicar fielmente la metodología del paper original
%   - Garantizar que la CNN aprenda mapeos físicamente realizables
%
% Ejemplo:
%   fmdl_16 = crear_modelo_fem(16);
%   img = mk_image(fmdl_16, 1.0);
%   volt_data = fwd_solve(img);
%   mediciones = extraer_96_mediciones(volt_data);
%   assert(length(mediciones) == 96);
%
% Ver también: extraer_40_mediciones, mapear_conductividad_8_a_16

%% Validación exhaustiva de entrada
if nargin ~= 1
   error('extraer_96_mediciones:WrongNumberOfInputs', ...
       'La función requiere exactamente 1 argumento de entrada');
end

% Validar que volt_data_16e es una estructura
if ~isstruct(volt_data_16e)
   error('extraer_96_mediciones:InvalidInput', ...
       'volt_data_16e debe ser una estructura retornada por fwd_solve de EIDORS. Tipo recibido: %s', ...
       class(volt_data_16e));
end

%% Búsqueda robusta del campo de mediciones
% EIDORS puede usar diferentes nombres de campo según la versión
possible_fields = {'meas', 'volt', 'voltage', 'measurements', 'data'};
mediciones_raw = [];
field_found = '';

for i = 1:length(possible_fields)
   field_name = possible_fields{i};
   if isfield(volt_data_16e, field_name)
       mediciones_raw = volt_data_16e.(field_name);
       field_found = field_name;
       break;
   end
end

% Verificar que se encontró un campo válido
if isempty(mediciones_raw)
   available_fields = fieldnames(volt_data_16e);
   error('extraer_96_mediciones:NoMeasurementsField', ...
       'No se encontró campo de mediciones en volt_data_16e.\nCampos disponibles: %s\nCampos buscados: %s', ...
       strjoin(available_fields, ', '), strjoin(possible_fields, ', '));
end

%% Validación de datos extraídos
% Verificar que no está vacío
if isempty(mediciones_raw)
   error('extraer_96_mediciones:EmptyMeasurements', ...
       'El campo de mediciones "%s" está vacío', field_found);
end

% Verificar que es numérico
if ~isnumeric(mediciones_raw)
   error('extraer_96_mediciones:NonNumericMeasurements', ...
       'Las mediciones en el campo "%s" deben ser numéricas. Tipo encontrado: %s', ...
       field_found, class(mediciones_raw));
end

% Convertir a vector columna
mediciones_vector = mediciones_raw(:);
n_mediciones_total = length(mediciones_vector);

%% Verificación de valores finitos
non_finite_mask = ~isfinite(mediciones_vector);
n_non_finite = sum(non_finite_mask);

if n_non_finite > 0
   warning('extraer_96_mediciones:NonFiniteValues', ...
       'Se encontraron %d valores no finitos en las mediciones (%.1f%% del total)', ...
       n_non_finite, (n_non_finite/n_mediciones_total)*100);
   
   % Reemplazar valores no finitos con ceros
   mediciones_vector(non_finite_mask) = 0;
   warning('extraer_96_mediciones:NonFiniteReplaced', ...
       'Valores no finitos reemplazados con ceros para mantener estabilidad');
end

%% Validación del número de mediciones para sistema de 16 electrodos
n_expected = 208;  % 16 excitaciones × 13 mediciones por excitación

if n_mediciones_total < n_expected
   error('extraer_96_mediciones:InsufficientMeasurements', ...
       'Insuficientes mediciones para sistema de 16 electrodos.\nEncontradas: %d, Esperadas: %d (mínimo)\nVerifique que el modelo FEM sea realmente de 16 electrodos.', ...
       n_mediciones_total, n_expected);
       
elseif n_mediciones_total > n_expected
   warning('extraer_96_mediciones:ExcessMeasurements', ...
       'Se encontraron %d mediciones, se esperaban %d. Usando las primeras %d.', ...
       n_mediciones_total, n_expected, n_expected);
   
   % Truncar a las primeras 208 mediciones
   mediciones_vector = mediciones_vector(1:n_expected);
   n_mediciones_total = n_expected;
end

%% IMPLEMENTACIÓN DE LA METODOLOGÍA XU ET AL. (2022)
% Parámetros de la metodología de canales virtuales
n_electrodos = 16;
n_excitaciones_totales = 16;
n_mediciones_por_excitacion = 13;
n_mediciones_virtuales_por_excitacion = 12;  % Tomar 12 de 13

% Excitaciones alternadas a seleccionar: 1, 3, 5, 7, 9, 11, 13, 15
excitaciones_alternadas = 1:2:n_electrodos;  % [1, 3, 5, 7, 9, 11, 13, 15]
n_excitaciones_virtuales = length(excitaciones_alternadas);  % 8

% Verificar que tenemos el número correcto de excitaciones virtuales
if n_excitaciones_virtuales ~= 8
   error('extraer_96_mediciones:WrongVirtualExcitationCount', ...
       'Error interno: se esperaban 8 excitaciones virtuales, calculadas: %d', ...
       n_excitaciones_virtuales);
end

%% Validación de la estructura de datos
% Verificar que el número total de mediciones es consistente
if n_mediciones_total ~= n_excitaciones_totales * n_mediciones_por_excitacion
   error('extraer_96_mediciones:InconsistentDataStructure', ...
       'Estructura de datos inconsistente.\nMediciones totales: %d\nEsperadas: %d excitaciones × %d mediciones = %d', ...
       n_mediciones_total, n_excitaciones_totales, n_mediciones_por_excitacion, ...
       n_excitaciones_totales * n_mediciones_por_excitacion);
end

%% Extracción según metodología de canales virtuales
% Inicializar vector de salida
mediciones_96 = zeros(96, 1);
idx_salida = 1;

% Información para debugging
extraction_info = struct();
extraction_info.excitaciones_seleccionadas = excitaciones_alternadas;
extraction_info.mediciones_por_excitacion_original = n_mediciones_por_excitacion;
extraction_info.mediciones_tomadas_por_excitacion = n_mediciones_virtuales_por_excitacion;
extraction_info.total_excitaciones_virtuales = n_excitaciones_virtuales;

% Activar debugging detallado si está habilitado
debug_enabled = false;
try
   debug_enabled = evalin('caller', 'exist(''debug_extraer_mediciones'', ''var'') && debug_extraer_mediciones');
catch
   % Si hay error evaluando, asumir que debugging está deshabilitado
end

if debug_enabled
   fprintf('\n=== EXTRACCIÓN 96 MEDICIONES - METODOLOGÍA XU ET AL. ===\n');
   fprintf('Mediciones totales disponibles: %d\n', n_mediciones_total);
   fprintf('Estructura: %d excitaciones × %d mediciones\n', n_excitaciones_totales, n_mediciones_por_excitacion);
   fprintf('Excitaciones seleccionadas: %s\n', mat2str(excitaciones_alternadas));
   fprintf('Mediciones por excitación virtual: %d de %d\n', ...
       n_mediciones_virtuales_por_excitacion, n_mediciones_por_excitacion);
   fprintf('Target final: %d mediciones\n', n_excitaciones_virtuales * n_mediciones_virtuales_por_excitacion);
   fprintf('========================================================\n');
end

% Procesar cada excitación alternada seleccionada
for i = 1:n_excitaciones_virtuales
   excitacion_actual = excitaciones_alternadas(i);
   
   % Calcular índices de inicio y fin para esta excitación en el vector original
   % Excitación k tiene mediciones en posiciones: (k-1)*13 + 1 hasta k*13
   inicio_excitacion = (excitacion_actual - 1) * n_mediciones_por_excitacion + 1;
   fin_mediciones_completas = inicio_excitacion + n_mediciones_por_excitacion - 1;
   
   % Tomar solo las primeras 12 mediciones de esta excitación
   fin_mediciones_virtuales = inicio_excitacion + n_mediciones_virtuales_por_excitacion - 1;
   
   % Verificar que los índices están dentro del rango válido
   if fin_mediciones_virtuales > n_mediciones_total
       error('extraer_96_mediciones:IndexOutOfRange', ...
           'Error en el cálculo de índices para excitación %d.\nÍndice final calculado: %d, Máximo disponible: %d', ...
           excitacion_actual, fin_mediciones_virtuales, n_mediciones_total);
   end
   
   % Extraer las 12 mediciones de esta excitación virtual
   indices_fuente = inicio_excitacion:fin_mediciones_virtuales;
   n_mediciones_extraidas = length(indices_fuente);
   
   % Calcular índices de destino en el vector de salida
   indices_destino = idx_salida:(idx_salida + n_mediciones_extraidas - 1);
   
   % Verificar que no excedemos el vector de salida
   if max(indices_destino) > 96
       error('extraer_96_mediciones:OutputIndexOutOfRange', ...
           'Error interno: índice de salida fuera de rango.\nÍndice máximo: %d, Límite: 96', ...
           max(indices_destino));
   end
   
   % Copiar las mediciones al vector de salida
   mediciones_96(indices_destino) = mediciones_vector(indices_fuente);
   
   % Información de debugging para esta excitación
   if debug_enabled
       fprintf('Excitación virtual %d (física %2d): índices %3d-%3d → posiciones %2d-%2d (%d mediciones)\n', ...
           i, excitacion_actual, inicio_excitacion, fin_mediciones_virtuales, ...
           idx_salida, max(indices_destino), n_mediciones_extraidas);
   end
   
   % Actualizar índice de salida para la siguiente excitación
   idx_salida = idx_salida + n_mediciones_extraidas;
end

%% Validación final crítica
% Verificar que se extrajeron exactamente 96 mediciones
n_extraidas = idx_salida - 1;
if n_extraidas ~= 96
   error('extraer_96_mediciones:WrongOutputCount', ...
       'Error crítico: se extrajeron %d mediciones en lugar de 96', n_extraidas);
end

% Verificar dimensión del vector de salida
if length(mediciones_96) ~= 96
   error('extraer_96_mediciones:WrongOutputSize', ...
       'Error interno: vector de salida tiene %d elementos en lugar de 96', length(mediciones_96));
end

% Asegurar que es vector columna
if isrow(mediciones_96)
   mediciones_96 = mediciones_96(:);
end

% Verificar que todos los valores finales son finitos
final_non_finite = sum(~isfinite(mediciones_96));
if final_non_finite > 0
   error('extraer_96_mediciones:NonFiniteOutput', ...
       'Error crítico: %d valores no finitos en la salida final', final_non_finite);
end

%% Análisis estadístico de la salida
mediciones_stats = struct();
mediciones_stats.min_value = min(mediciones_96);
mediciones_stats.max_value = max(mediciones_96);
mediciones_stats.mean_value = mean(mediciones_96);
mediciones_stats.std_value = std(mediciones_96);
mediciones_stats.range = mediciones_stats.max_value - mediciones_stats.min_value;

% Detectar posibles anomalías
if mediciones_stats.max_value > 1e6
   warning('extraer_96_mediciones:LargeValues', ...
       'Se detectaron valores muy grandes en las mediciones (máx: %.2e)', ...
       mediciones_stats.max_value);
end

if mediciones_stats.range < 1e-12
   warning('extraer_96_mediciones:ConstantValues', ...
       'Las mediciones parecen ser casi constantes (rango: %.2e)', ...
       mediciones_stats.range);
end

%% Información final de debugging
if debug_enabled
   fprintf('\n=== RESULTADOS FINALES ===\n');
   fprintf('Mediciones extraídas: %d\n', length(mediciones_96));
   fprintf('Rango: [%.6e, %.6e]\n', mediciones_stats.min_value, mediciones_stats.max_value);
   fprintf('Media ± Desv: %.6e ± %.6e\n', mediciones_stats.mean_value, mediciones_stats.std_value);
   fprintf('Metodología: Xu et al. (2022) - Virtual Channel Based DA\n');
   fprintf('Patrón: %d excitaciones alternadas × %d mediciones = %d total\n', ...
       n_excitaciones_virtuales, n_mediciones_virtuales_por_excitacion, ...
       n_excitaciones_virtuales * n_mediciones_virtuales_por_excitacion);
   fprintf('Data leakage: EVITADO mediante extracción sistemática\n');
   fprintf('==========================\n\n');
end

end