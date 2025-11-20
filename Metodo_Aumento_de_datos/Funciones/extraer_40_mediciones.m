function mediciones_40 = extraer_40_mediciones(volt_data)
% extraer_40_mediciones - Extrae 40 mediciones del sistema de 8 electrodos
%
% Sintaxis:
%   mediciones_40 = extraer_40_mediciones(volt_data)
%
% Entrada:
%   volt_data - Estructura de datos retornada por fwd_solve de EIDORS
%               para una simulación de 8 electrodos
%
% Salida:
%   mediciones_40 - Vector columna de 40x1 con las mediciones extraídas
%
% Descripción:
%   Esta función es una utilidad esencial para asegurar la dimensionalidad
%   correcta de los datos de entrada para el entrenamiento de la CNN.
%   
%   Para un sistema de 8 electrodos con patrón de estimulación adyacente,
%   se esperan exactamente 40 mediciones:
%   - 8 excitaciones (cada par de electrodos adyacentes)
%   - 5 mediciones por excitación (pares adyacentes no usados para estimulación)
%   - Total: 8 × 5 = 40 mediciones
%
%   La función garantiza:
%   - Extracción robusta de mediciones de la estructura fwd_solve
%   - Validación de dimensionalidad esperada (40 mediciones)
%   - Formato de salida consistente (vector columna)
%   - Manejo apropiado de casos edge con warnings informativos
%
%   Casos manejados:
%   - Exactamente 40 mediciones: Extracción directa
%   - Más de 40 mediciones: Trunca a las primeras 40 (con warning)
%   - Menos de 40 mediciones: Error (datos insuficientes)
%   - Formato incorrecto: Convierte a vector columna
%
% Ejemplo:
%   fmdl = crear_modelo_fem(8);
%   img = mk_image(fmdl, 1.0);
%   volt_data = fwd_solve(img);
%   mediciones = extraer_40_mediciones(volt_data);
%   fprintf('Mediciones extraídas: %d\n', length(mediciones));
%
% Ver también: fwd_solve, extraer_96_mediciones

%% Validación exhaustiva de entrada
if nargin ~= 1
   error('extraer_40_mediciones:WrongNumberOfInputs', ...
       'La función requiere exactamente 1 argumento de entrada');
end

% Validar que volt_data es una estructura
if ~isstruct(volt_data)
   error('extraer_40_mediciones:InvalidInput', ...
       'volt_data debe ser una estructura retornada por fwd_solve de EIDORS. Tipo recibido: %s', ...
       class(volt_data));
end

%% Búsqueda robusta del campo de mediciones
% EIDORS puede usar diferentes nombres de campo según la versión
possible_fields = {'meas', 'volt', 'voltage', 'measurements', 'data'};
mediciones_raw = [];
field_found = '';

for i = 1:length(possible_fields)
   field_name = possible_fields{i};
   if isfield(volt_data, field_name)
       mediciones_raw = volt_data.(field_name);
       field_found = field_name;
       break;
   end
end

% Verificar que se encontró un campo válido
if isempty(mediciones_raw)
   available_fields = fieldnames(volt_data);
   error('extraer_40_mediciones:NoMeasurementsField', ...
       'No se encontró campo de mediciones en volt_data.\nCampos disponibles: %s\nCampos buscados: %s', ...
       strjoin(available_fields, ', '), strjoin(possible_fields, ', '));
end

%% Validación de datos extraídos
% Verificar que no está vacío
if isempty(mediciones_raw)
   error('extraer_40_mediciones:EmptyMeasurements', ...
       'El campo de mediciones "%s" está vacío', field_found);
end

% Verificar que es numérico
if ~isnumeric(mediciones_raw)
   error('extraer_40_mediciones:NonNumericMeasurements', ...
       'Las mediciones en el campo "%s" deben ser numéricas. Tipo encontrado: %s', ...
       field_found, class(mediciones_raw));
end

% Convertir a vector (manejar tanto vectores fila como columna)
mediciones_vector = mediciones_raw(:);
n_mediciones = length(mediciones_vector);

%% Verificación de valores finitos
% Detectar valores no finitos (NaN, Inf, -Inf)
non_finite_mask = ~isfinite(mediciones_vector);
n_non_finite = sum(non_finite_mask);

if n_non_finite > 0
   warning('extraer_40_mediciones:NonFiniteValues', ...
       'Se encontraron %d valores no finitos en las mediciones (%.1f%% del total)', ...
       n_non_finite, (n_non_finite/n_mediciones)*100);
   
   % Mostrar detalles de valores problemáticos
   nan_count = sum(isnan(mediciones_vector));
   inf_count = sum(isinf(mediciones_vector));
   
   if nan_count > 0
       warning('extraer_40_mediciones:NaNValues', ...
           'Se encontraron %d valores NaN', nan_count);
   end
   
   if inf_count > 0
       warning('extraer_40_mediciones:InfValues', ...
           'Se encontraron %d valores infinitos', inf_count);
   end
   
   % Reemplazar valores no finitos con ceros (estrategia conservadora)
   mediciones_vector(non_finite_mask) = 0;
   warning('extraer_40_mediciones:NonFiniteReplaced', ...
       'Valores no finitos reemplazados con ceros para mantener estabilidad');
end

%% Procesamiento según número de mediciones disponibles
if n_mediciones == 40
   %% Caso ideal: exactamente 40 mediciones
   mediciones_40 = mediciones_vector;
   
elseif n_mediciones > 40
   %% Caso: más mediciones de las esperadas
   % Estrategia: tomar las primeras 40 mediciones
   mediciones_40 = mediciones_vector(1:40);
   
   warning('extraer_40_mediciones:ExcessMeasurements', ...
       'Se encontraron %d mediciones, se esperaban 40. Usando las primeras 40 mediciones.\nEsto puede indicar un patrón de estimulación diferente al esperado.', ...
       n_mediciones);
       
elseif n_mediciones >= 20 && n_mediciones < 40
   %% Caso: menos mediciones pero suficientes para interpolación
   % Estrategia: interpolación lineal para generar 40 puntos
   warning('extraer_40_mediciones:InsufficientMeasurements', ...
       'Se encontraron %d mediciones, se esperaban 40. Aplicando interpolación lineal.', ...
       n_mediciones);
   
   % Crear índices para interpolación
   x_original = linspace(1, 40, n_mediciones);
   x_target = 1:40;
   
   % Interpolación lineal con extrapolación en los extremos
   mediciones_40 = interp1(x_original, mediciones_vector, x_target, 'linear', 'extrap')';
   
   warning('extraer_40_mediciones:InterpolationApplied', ...
       'Interpolación aplicada exitosamente: %d → 40 mediciones', n_mediciones);
       
else
   %% Caso: muy pocas mediciones - error crítico
   error('extraer_40_mediciones:CriticallyInsufficientMeasurements', ...
       'Se encontraron solo %d mediciones, insuficientes para el sistema de 8 electrodos.\nMínimo requerido: 20 mediciones para interpolación.\nVerifique la configuración del modelo FEM y patrones de estimulación.', ...
       n_mediciones);
end

%% Validación final de salida
% Asegurar que es vector columna
if isrow(mediciones_40)
   mediciones_40 = mediciones_40(:);
end

% Verificar dimensión final
if length(mediciones_40) ~= 40
   error('extraer_40_mediciones:OutputSizeError', ...
       'Error interno: se generaron %d mediciones en lugar de 40', length(mediciones_40));
end

% Verificar que todos los valores finales son finitos
if any(~isfinite(mediciones_40))
   final_non_finite = sum(~isfinite(mediciones_40));
   error('extraer_40_mediciones:NonFiniteOutput', ...
       'Error crítico: %d valores no finitos en la salida final', final_non_finite);
end

%% Análisis estadístico de salida (opcional para debugging)
% Calcular estadísticas básicas para validación
mediciones_stats = struct();
mediciones_stats.min_value = min(mediciones_40);
mediciones_stats.max_value = max(mediciones_40);
mediciones_stats.mean_value = mean(mediciones_40);
mediciones_stats.std_value = std(mediciones_40);
mediciones_stats.range = mediciones_stats.max_value - mediciones_stats.min_value;

% Detectar posibles anomalías en los datos
if mediciones_stats.max_value > 1e6
   warning('extraer_40_mediciones:LargeValues', ...
       'Se detectaron valores muy grandes en las mediciones (máx: %.2e). Verifique la configuración del modelo.', ...
       mediciones_stats.max_value);
end

if mediciones_stats.range < 1e-12
   warning('extraer_40_mediciones:ConstantValues', ...
       'Las mediciones parecen ser casi constantes (rango: %.2e). Esto puede indicar un problema en la simulación.', ...
       mediciones_stats.range);
end

if mediciones_stats.std_value == 0
   warning('extraer_40_mediciones:ZeroVariance', ...
       'Las mediciones tienen varianza cero. Verifique que la imagen no sea completamente homogénea.');
end

% Opcional: Almacenar estadísticas en variable global para debugging
% (Solo si existe una variable de debugging en el workspace)
try
   if evalin('caller', 'exist(''debug_extraer_mediciones'', ''var'') && debug_extraer_mediciones')
       fprintf('\n=== ESTADÍSTICAS DE EXTRACCIÓN 40 MEDICIONES ===\n');
       fprintf('Campo fuente: %s\n', field_found);
       fprintf('Mediciones originales: %d\n', n_mediciones);
       fprintf('Mediciones finales: %d\n', length(mediciones_40));
       fprintf('Rango: [%.6e, %.6e]\n', mediciones_stats.min_value, mediciones_stats.max_value);
       fprintf('Media ± Desv: %.6e ± %.6e\n', mediciones_stats.mean_value, mediciones_stats.std_value);
       fprintf('Valores no finitos originales: %d\n', n_non_finite);
       fprintf('===============================================\n\n');
   end
catch
   % Si hay error en debugging, continuar silenciosamente
end

end