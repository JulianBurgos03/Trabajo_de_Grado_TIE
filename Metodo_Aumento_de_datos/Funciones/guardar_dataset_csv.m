function guardar_dataset_csv(X_data, y_data, filename)
% guardar_dataset_csv - Guarda dataset EIT en formato CSV con headers descriptivos
%
% Sintaxis:
%   guardar_dataset_csv(X_data, y_data, filename)
%
% Entradas:
%   X_data   - Matriz de datos de entrada (N × 40) - sistema 8 electrodos
%   y_data   - Matriz de datos de salida (N × 96) - sistema 16 electrodos
%   filename - Nombre del archivo CSV a crear (char o string)
%
% Salida:
%   Ninguna (guarda archivo en disco y muestra confirmación)
%
% Descripción:
%   Esta función es la utilidad final del pipeline de generación de dataset.
%   Combina las matrices de entrada y salida del dataset EIT, genera headers
%   descriptivos y guarda todo en formato CSV estándar compatible con
%   herramientas de machine learning.
%
%   PROPÓSITO CRÍTICO:
%   Crear el archivo final que será usado para entrenar la CNN en Python.
%   El formato debe ser compatible con pandas, scikit-learn y TensorFlow/Keras.
%
%   ESTRUCTURA DEL ARCHIVO CSV:
%   - Columnas 1-40:   med_8e_01, med_8e_02, ..., med_8e_40 (entrada X)
%   - Columnas 41-136: med_16e_01, med_16e_02, ..., med_16e_96 (salida y)
%   - Total: 136 columnas × N muestras filas
%   - Primera fila: Headers descriptivos
%   - Filas 2 en adelante: Datos numéricos
%
%   FORMATO DE HEADERS:
%   med_8e_XX  = Medición XX del sistema de 8 electrodos (entrada CNN)
%   med_16e_XX = Medición XX del sistema de 16 electrodos (ground truth)
%
%   COMPATIBILIDAD:
%   El archivo generado puede ser leído directamente en Python con:
%   import pandas as pd
%   data = pd.read_csv('filename.csv')
%   X = data.iloc[:, :40].values      # Entrada (8 electrodos)
%   y = data.iloc[:, 40:].values      # Salida (16 electrodos)
%
% Ejemplo:
%   % Ejemplo con datos sintéticos
%   N_samples = 1000;
%   X_data = randn(N_samples, 40);   % 1000 muestras de entrada
%   y_data = randn(N_samples, 96);   % 1000 muestras de salida
%   guardar_dataset_csv(X_data, y_data, 'dataset_eit.csv');
%
% Ver también: array2table, writetable, readtable

%% Validación exhaustiva de entradas
if nargin ~= 3
   error('guardar_dataset_csv:WrongNumberOfInputs', ...
       'La función requiere exactamente 3 argumentos de entrada');
end

% Validar matrices de datos
if ~isnumeric(X_data) || ~ismatrix(X_data)
   error('guardar_dataset_csv:InvalidXData', ...
       'X_data debe ser una matriz numérica. Tipo recibido: %s', class(X_data));
end

if ~isnumeric(y_data) || ~ismatrix(y_data)
   error('guardar_dataset_csv:InvalidYData', ...
       'y_data debe ser una matriz numérica. Tipo recibido: %s', class(y_data));
end

% Verificar dimensiones específicas esperadas
[N_samples_X, N_features_X] = size(X_data);
[N_samples_y, N_features_y] = size(y_data);

% Validar dimensiones de características
if N_features_X ~= 40
   error('guardar_dataset_csv:WrongXDimensions', ...
       'X_data debe tener exactamente 40 columnas (mediciones 8e). Encontradas: %d', N_features_X);
end

if N_features_y ~= 96
   error('guardar_dataset_csv:WrongYDimensions', ...
       'y_data debe tener exactamente 96 columnas (mediciones 16e). Encontradas: %d', N_features_y);
end

% Verificar consistencia en número de muestras
if N_samples_X ~= N_samples_y
   error('guardar_dataset_csv:SampleCountMismatch', ...
       'X_data (%d muestras) e y_data (%d muestras) deben tener el mismo número de filas', ...
       N_samples_X, N_samples_y);
end

N_samples = N_samples_X;  % Número total de muestras

% Validar nombre de archivo
if ~(ischar(filename) || isstring(filename))
   error('guardar_dataset_csv:InvalidFilename', ...
       'filename debe ser un char array o string. Tipo recibido: %s', class(filename));
end

filename = char(filename);  % Convertir a char para compatibilidad

% Verificar y añadir extensión .csv si es necesario
if ~endsWith(lower(filename), '.csv')
   warning('guardar_dataset_csv:NoCSVExtension', ...
       'El archivo no tiene extensión .csv. Añadiendo automáticamente.');
   filename = [filename, '.csv'];
end

%% Análisis de calidad de datos
fprintf('📊 ANÁLISIS DE CALIDAD DE DATOS...\n');

% Verificar valores finitos en X_data
non_finite_X = sum(~isfinite(X_data(:)));
if non_finite_X > 0
   warning('guardar_dataset_csv:NonFiniteXData', ...
       'X_data contiene %d valores no finitos (%.3f%% del total)', ...
       non_finite_X, (non_finite_X/numel(X_data))*100);
end

% Verificar valores finitos en y_data
non_finite_y = sum(~isfinite(y_data(:)));
if non_finite_y > 0
   warning('guardar_dataset_csv:NonFiniteYData', ...
       'y_data contiene %d valores no finitos (%.3f%% del total)', ...
       non_finite_y, (non_finite_y/numel(y_data))*100);
end

% Mostrar estadísticas descriptivas
fprintf('   📈 X_data (8e):  [%.3e, %.3e], media=%.3e, std=%.3e\n', ...
   min(X_data(:)), max(X_data(:)), mean(X_data(:)), std(X_data(:)));
fprintf('   📈 y_data (16e): [%.3e, %.3e], media=%.3e, std=%.3e\n', ...
   min(y_data(:)), max(y_data(:)), mean(y_data(:)), std(y_data(:)));

% Detectar posibles problemas en los datos
if std(X_data(:)) == 0
   warning('guardar_dataset_csv:ZeroVarianceX', ...
       'X_data tiene varianza cero - todos los valores son idénticos');
end

if std(y_data(:)) == 0
   warning('guardar_dataset_csv:ZeroVarianceY', ...
       'y_data tiene varianza cero - todos los valores son idénticos');
end

%% Generación programática de headers descriptivos
fprintf('📝 GENERANDO HEADERS DESCRIPTIVOS...\n');

% Headers para datos de 8 electrodos (columnas 1-40)
headers_8e = cell(1, 40);
for i = 1:40
   headers_8e{i} = sprintf('med_8e_%02d', i);
end

% Headers para datos de 16 electrodos (columnas 41-136)
headers_16e = cell(1, 96);
for i = 1:96
   headers_16e{i} = sprintf('med_16e_%02d', i);
end

% Combinar todos los headers en orden correcto
headers_combined = [headers_8e, headers_16e];
N_total_features = length(headers_combined);

% Verificar que tenemos el número correcto de headers
if N_total_features ~= 136
   error('guardar_dataset_csv:WrongHeaderCount', ...
       'Error interno: se generaron %d headers en lugar de 136', N_total_features);
end

fprintf('   ✅ Headers generados: %d para 8e + %d para 16e = %d total\n', ...
   length(headers_8e), length(headers_16e), N_total_features);

%% Combinación horizontal de matrices de datos
fprintf('🔗 COMBINANDO MATRICES DE DATOS...\n');

% Concatenar X_data e y_data horizontalmente
combined_data = [X_data, y_data];

% Verificar dimensiones de la matriz combinada
[final_rows, final_cols] = size(combined_data);
fprintf('   📊 Matriz combinada: %d muestras × %d características\n', final_rows, final_cols);

% Validar dimensiones finales
if final_cols ~= N_total_features
   error('guardar_dataset_csv:DimensionMismatch', ...
       'Error crítico: matriz combinada (%d cols) vs headers (%d cols)', ...
       final_cols, N_total_features);
end

if final_rows ~= N_samples
   error('guardar_dataset_csv:SampleCountCorruption', ...
       'Error crítico: número de muestras cambió durante la combinación');
end

%% Conversión a tabla de MATLAB y guardado
fprintf('💾 GUARDANDO ARCHIVO CSV: %s\n', filename);

try
   % Crear tabla con headers descriptivos
   fprintf('   📋 Creando tabla con headers...\n');
   tic;
   dataset_table = array2table(combined_data, 'VariableNames', headers_combined);
   table_creation_time = toc;
   
   fprintf('   ⏱️  Tiempo creación tabla: %.2f segundos\n', table_creation_time);
   
   % Guardar tabla usando writetable
   fprintf('   💾 Escribiendo archivo CSV...\n');
   tic;
   writetable(dataset_table, filename);
   file_write_time = toc;
   
   fprintf('   ⏱️  Tiempo escritura archivo: %.2f segundos\n', file_write_time);
   
catch ME
   % Si writetable falla, intentar método manual de respaldo
   warning('guardar_dataset_csv:WritetableFailed', ...
       'writetable falló: %s. Intentando método manual...', ME.message);
   
   try
       fprintf('   📝 Usando método de escritura manual...\n');
       tic;
       guardar_csv_manual(combined_data, headers_combined, filename);
       manual_write_time = toc;
       
       fprintf('   ⏱️  Tiempo escritura manual: %.2f segundos\n', manual_write_time);
       fprintf('   ✅ Archivo guardado con método manual\n');
       
   catch ME2
       error('guardar_dataset_csv:BothMethodsFailed', ...
           'Ambos métodos fallaron.\nwritetable: %s\nManual: %s', ...
           ME.message, ME2.message);
   end
end

%% Verificación del archivo creado
fprintf('🔍 VERIFICANDO ARCHIVO CREADO...\n');

% Obtener información del archivo
try
   file_info = dir(filename);
   if isempty(file_info)
       error('guardar_dataset_csv:FileNotCreated', ...
           'El archivo no se creó correctamente: %s', filename);
   end
   
   file_size_bytes = file_info.bytes;
   file_size_mb = file_size_bytes / (1024^2);
   
   fprintf('   📁 Archivo: %s\n', filename);
   fprintf('   📏 Tamaño: %.2f MB (%s bytes)\n', file_size_mb, addcomma(file_size_bytes));
   fprintf('   📅 Fecha: %s\n', file_info.date);
   
   % Estimar tamaño esperado (aproximadamente)
   expected_size_approx = N_samples * N_total_features * 12; % ~12 bytes por valor
   size_ratio = file_size_bytes / expected_size_approx;
   
   if size_ratio < 0.3 || size_ratio > 3.0
       warning('guardar_dataset_csv:UnexpectedFileSize', ...
           'Tamaño de archivo inesperado. Ratio: %.2f (esperado ~1.0)', size_ratio);
   else
       fprintf('   ✅ Tamaño de archivo apropiado (ratio: %.2f)\n', size_ratio);
   end
   
catch ME
   warning('guardar_dataset_csv:VerificationFailed', ...
       'No se pudo verificar el archivo: %s', ME.message);
end

%% Validación de lectura rápida (opcional)
fprintf('🧪 VALIDACIÓN DE LECTURA...\n');

try
   % Intentar leer las primeras 5 filas para verificar formato
   test_table = readtable(filename, 'Range', '1:5');
   
   % Verificar dimensiones de la muestra
   [test_rows, test_cols] = size(test_table);
   fprintf('   📊 Test lectura: %d filas × %d columnas leídas\n', test_rows, test_cols);
   
   if test_cols ~= N_total_features
       warning('guardar_dataset_csv:ReadbackDimensionMismatch', ...
           'Dimensiones de lectura (%d cols) difieren de las esperadas (%d cols)', ...
           test_cols, N_total_features);
   else
       fprintf('   ✅ Dimensiones de lectura correctas\n');
   end
   
   % Verificar algunos headers
   headers_read = test_table.Properties.VariableNames;
   if strcmp(headers_read{1}, 'med_8e_01') && strcmp(headers_read{end}, 'med_16e_96')
       fprintf('   ✅ Headers verificados correctamente\n');
   else
       warning('guardar_dataset_csv:HeaderMismatch', ...
           'Headers no coinciden: primero="%s", último="%s"', ...
           headers_read{1}, headers_read{end});
   end
   
catch ME
   warning('guardar_dataset_csv:ReadbackValidationFailed', ...
       'No se pudo validar la lectura: %s', ME.message);
end

%% Mensaje de confirmación final
fprintf('\n');
fprintf('🎉================================================================🎉\n');
fprintf('   DATASET GUARDADO EXITOSAMENTE\n');
fprintf('🎉================================================================🎉\n');
fprintf('\n');
fprintf('📋 RESUMEN DEL DATASET:\n');
fprintf('   📁 Archivo: %s\n', filename);
fprintf('   📊 Dimensiones: %s muestras × %d características\n', addcomma(N_samples), N_total_features);
fprintf('   🎯 Entrada (8e): %d características (med_8e_01 a med_8e_40)\n', length(headers_8e));
fprintf('   🎯 Salida (16e): %d características (med_16e_01 a med_16e_96)\n', length(headers_16e));
fprintf('   💾 Formato: CSV con headers descriptivos\n');
fprintf('   🚀 Listo para entrenamiento de CNN\n');
fprintf('\n');
fprintf('💡 COMANDO PYTHON SUGERIDO:\n');
fprintf('   import pandas as pd\n');
fprintf('   data = pd.read_csv(''%s'')\n', filename);
fprintf('   X = data.iloc[:, :40].values   # Entrada (8 electrodos)\n');
fprintf('   y = data.iloc[:, 40:].values   # Salida (16 electrodos)\n');
fprintf('\n');
fprintf('================================================================\n');

end

%% =====================================================================
%% FUNCIÓN AUXILIAR: Método manual de escritura CSV
%% =====================================================================
function guardar_csv_manual(data, headers, filename)
% Método manual para guardar CSV cuando writetable falla

[N_rows, N_cols] = size(data);

% Abrir archivo para escritura
fid = fopen(filename, 'w');
if fid == -1
   error('guardar_csv_manual:CannotOpenFile', ...
       'No se puede abrir el archivo para escritura: %s', filename);
end

try
   % Escribir línea de headers
   header_line = strjoin(headers, ',');
   fprintf(fid, '%s\n', header_line);
   
   % Escribir datos fila por fila con barra de progreso
   fprintf('   📝 Escribiendo %s filas...\n', addcomma(N_rows));
   
   progress_interval = max(1, floor(N_rows / 20));  % Mostrar 20 actualizaciones
   
   for i = 1:N_rows
       % Crear string de la fila con formato científico
       row_values = data(i, :);
       row_strings = arrayfun(@(x) sprintf('%.8e', x), row_values, 'UniformOutput', false);
       row_line = strjoin(row_strings, ',');
       
       % Escribir fila
       fprintf(fid, '%s\n', row_line);
       
       % Mostrar progreso
       if mod(i, progress_interval) == 0 || i == N_rows
           percent_complete = (i / N_rows) * 100;
           fprintf('      Progreso: %.1f%% (%s/%s filas)\n', ...
               percent_complete, addcomma(i), addcomma(N_rows));
       end
   end
   
finally
   % Cerrar archivo siempre
   fclose(fid);
end

end

%% =====================================================================
%% FUNCIÓN AUXILIAR: Formatear números con comas
%% =====================================================================
function formatted_str = addcomma(number)
% Añade comas como separadores de miles para legibilidad

if ~isnumeric(number) || ~isscalar(number)
   formatted_str = 'N/A';
   return;
end

str = sprintf('%.0f', number);
formatted_str = regexprep(str, '(\d)(?=(\d{3})+(?!\d))', '$1,');

end