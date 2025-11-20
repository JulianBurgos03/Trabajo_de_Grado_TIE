function img_out = add_circular_inclusion(img, x_center, y_center, radius, conductivity)
% add_circular_inclusion - Añade una inclusión circular a una imagen de EIDORS
%
% Sintaxis:
%   img_out = add_circular_inclusion(img, x_center, y_center, radius, conductivity)
%
% Entradas:
%   img         - Estructura de imagen de EIDORS (creada con mk_image)
%   x_center    - Coordenada X del centro de la inclusión circular
%   y_center    - Coordenada Y del centro de la inclusión circular
%   radius      - Radio de la inclusión circular
%   conductivity- Valor de conductividad a asignar dentro de la inclusión
%
% Salida:
%   img_out     - Estructura de imagen modificada con la inclusión añadida
%
% Descripción:
%   Esta función modifica una imagen de EIDORS añadiendo una inclusión circular.
%   Calcula el centroide de cada elemento de la malla FEM, determina cuáles están
%   dentro del círculo especificado, y les asigna el valor de conductividad dado.
%   Es una función de utilidad robusta que mantiene la integridad de la imagen.
%
%   Proceso detallado:
%   1. Valida todas las entradas exhaustivamente
%   2. Extrae información del forward model de la imagen
%   3. Calcula centroides de todos los elementos de la malla
%   4. Determina qué elementos están dentro del círculo
%   5. Asigna la nueva conductividad a esos elementos
%   6. Añade metadatos informativos para trazabilidad
%
% Ejemplo:
%   fmdl = crear_modelo_fem(8);
%   img = mk_image(fmdl, 1.0);  % Imagen homogénea con conductividad 1.0
%   img_modified = add_circular_inclusion(img, 0.2, 0.1, 0.15, 0.3);
%
% Ver también: mk_image, crear_modelo_fem, generar_imagen_conductividad

%% Validación exhaustiva de entradas
if nargin ~= 5
   error('add_circular_inclusion:WrongNumberOfInputs', ...
       'La función requiere exactamente 5 argumentos de entrada');
end

% Validar estructura de imagen de EIDORS
if ~isstruct(img)
   error('add_circular_inclusion:InvalidImageInput', ...
       'img debe ser una estructura de imagen de EIDORS');
end

if ~isfield(img, 'fwd_model')
   error('add_circular_inclusion:MissingFwdModel', ...
       'La imagen debe contener un campo fwd_model');
end

if ~isfield(img, 'elem_data')
   error('add_circular_inclusion:MissingElemData', ...
       'La imagen debe contener un campo elem_data');
end

% Validar parámetros numéricos de entrada
numeric_params = {x_center, y_center, radius, conductivity};
param_names = {'x_center', 'y_center', 'radius', 'conductivity'};

for i = 1:length(numeric_params)
   param_value = numeric_params{i};
   param_name = param_names{i};
   
   if ~isnumeric(param_value) || ~isscalar(param_value) || ~isfinite(param_value)
       error('add_circular_inclusion:InvalidNumericInput', ...
           'El parámetro %s debe ser un número escalar finito. Valor recibido: %s', ...
           param_name, mat2str(param_value));
   end
end

% Validaciones específicas de rangos
if radius <= 0
   error('add_circular_inclusion:InvalidRadius', ...
       'El radio debe ser un valor positivo. Valor recibido: %.6f', radius);
end

if conductivity <= 0
   error('add_circular_inclusion:InvalidConductivity', ...
       'La conductividad debe ser un valor positivo. Valor recibido: %.6f', conductivity);
end

% Warnings para valores sospechosos pero no erróneos
if radius > 0.5
   warning('add_circular_inclusion:LargeRadius', ...
       'Radio muy grande (%.3f) puede exceder el dominio del modelo', radius);
end

if abs(x_center) > 0.8 || abs(y_center) > 0.8
   warning('add_circular_inclusion:CenterNearBoundary', ...
       'Centro de inclusión (%.3f, %.3f) está muy cerca del borde del dominio', ...
       x_center, y_center);
end

%% Extraer y validar información del modelo FEM
fmdl = img.fwd_model;

% Verificar campos esenciales del forward model
required_fmdl_fields = {'nodes', 'elems'};
for i = 1:length(required_fmdl_fields)
   if ~isfield(fmdl, required_fmdl_fields{i})
       error('add_circular_inclusion:InvalidFwdModel', ...
           'El forward model debe contener el campo: %s', required_fmdl_fields{i});
   end
end

nodes = fmdl.nodes;
elems = fmdl.elems;

% Validar dimensiones de nodos y elementos
if size(nodes, 2) < 2
   error('add_circular_inclusion:InvalidNodeDimensions', ...
       'Los nodos deben tener al menos 2 coordenadas (X, Y). Dimensiones encontradas: %s', ...
       mat2str(size(nodes)));
end

if size(elems, 2) < 3
   error('add_circular_inclusion:InvalidElemDimensions', ...
       'Los elementos deben tener al menos 3 nodos (triángulos). Dimensiones encontradas: %s', ...
       mat2str(size(elems)));
end

n_nodes = size(nodes, 1);
n_elems = size(elems, 1);

% Validar consistencia entre elem_data y número de elementos
if length(img.elem_data) ~= n_elems
   error('add_circular_inclusion:ElemDataSizeMismatch', ...
       'El tamaño de elem_data (%d) no coincide con el número de elementos (%d)', ...
       length(img.elem_data), n_elems);
end

%% Crear copia de la imagen para modificar (preservar original)
img_out = img;

%% Cálculo robusto de centroides de elementos
centroids = zeros(n_elems, 2);

for i = 1:n_elems
   % Obtener índices de nodos del elemento actual
   node_indices = elems(i, :);
   
   % Validar que los índices de nodos están en rango válido
   if any(node_indices > n_nodes) || any(node_indices < 1)
       error('add_circular_inclusion:InvalidNodeIndices', ...
           'Índices de nodos fuera de rango en elemento %d. Índices: %s, Rango válido: [1, %d]', ...
           i, mat2str(node_indices), n_nodes);
   end
   
   % Verificar que no hay índices duplicados (elemento degenerado)
   if length(unique(node_indices)) ~= length(node_indices)
       warning('add_circular_inclusion:DegenerateElement', ...
           'Elemento %d tiene nodos duplicados: %s', i, mat2str(node_indices));
   end
   
   % Obtener coordenadas (x, y) de los nodos del elemento
   element_nodes = nodes(node_indices, 1:2);
   
   % Calcular centroide como promedio aritmético de las coordenadas
   centroids(i, 1) = mean(element_nodes(:, 1)); % Coordenada X
   centroids(i, 2) = mean(element_nodes(:, 2)); % Coordenada Y
end

%% Selección de elementos dentro de la inclusión circular
% Calcular distancias euclidianas desde cada centroide al centro de la inclusión
distances = sqrt((centroids(:, 1) - x_center).^2 + (centroids(:, 2) - y_center).^2);

% Crear vector lógico: true para elementos dentro del círculo
elementos_dentro = distances <= radius;

% Estadísticas de la selección
n_elementos_afectados = sum(elementos_dentro);
fraccion_afectada = n_elementos_afectados / n_elems;

%% Modificación de conductividad
if n_elementos_afectados > 0
   % Asignar nueva conductividad a elementos seleccionados
   img_out.elem_data(elementos_dentro) = conductivity;
else
   % Ningún elemento capturado - emitir warning informativo
   warning('add_circular_inclusion:NoElementsAffected', ...
       'Ningún elemento está dentro de la inclusión especificada.\n   Centro: (%.3f, %.3f), Radio: %.3f\n   Posible causa: inclusión muy pequeña para la resolución de la malla', ...
       x_center, y_center, radius);
end

%% Validación de integridad de la imagen modificada
% Verificar que no se introdujeron valores no finitos
if any(~isfinite(img_out.elem_data))
   error('add_circular_inclusion:NonFiniteValuesGenerated', ...
       'Se generaron valores no finitos en elem_data durante la modificación');
end

% Verificar que el tamaño de elem_data se mantiene
if length(img_out.elem_data) ~= n_elems
   error('add_circular_inclusion:DataSizeCorruption', ...
       'Error crítico: el tamaño de elem_data cambió durante la modificación');
end

%% Añadir metadatos informativos para trazabilidad
% Inicializar historial de inclusiones si no existe
if ~isfield(img_out, 'inclusion_history')
   img_out.inclusion_history = [];
end

% Crear registro detallado de esta inclusión
inclusion_record = struct();
inclusion_record.x_center = x_center;
inclusion_record.y_center = y_center;
inclusion_record.radius = radius;
inclusion_record.conductivity = conductivity;
inclusion_record.elements_affected = n_elementos_afectados;
inclusion_record.total_elements = n_elems;
inclusion_record.fraction_affected = fraccion_afectada;
inclusion_record.min_distance_to_center = min(distances);
inclusion_record.max_distance_to_center = max(distances);
inclusion_record.timestamp = datestr(now);

% Añadir al historial
img_out.inclusion_history = [img_out.inclusion_history, inclusion_record];

% Actualizar contador total de inclusiones
img_out.total_inclusions_added = length(img_out.inclusion_history);

%% Calcular y actualizar estadísticas globales de conductividad
img_out.conductivity_stats = struct();
img_out.conductivity_stats.min_value = min(img_out.elem_data);
img_out.conductivity_stats.max_value = max(img_out.elem_data);
img_out.conductivity_stats.mean_value = mean(img_out.elem_data);
img_out.conductivity_stats.std_value = std(img_out.elem_data);
img_out.conductivity_stats.unique_values = length(unique(img_out.elem_data));

end