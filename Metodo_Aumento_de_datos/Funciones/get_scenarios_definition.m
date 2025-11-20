function scenarios = get_scenarios_definition()
% get_scenarios_definition - Define el catálogo completo de phantoms EIT
%
% Sintaxis:
%   scenarios = get_scenarios_definition()
%
% Entrada:
%   Ninguna
%
% Salida:
%   scenarios - Cell array de 1x10 con structs que definen cada escenario
%
% Descripción:
%   Esta función actúa como base de datos central de phantoms para las 
%   simulaciones EIT. Define 10 escenarios predefinidos que van desde 
%   configuraciones simples (homogéneo, una inclusión) hasta configuraciones 
%   complejas (múltiples inclusiones en diferentes disposiciones geométricas).
%
%   Estructura de cada escenario:
%   - name: Nombre descriptivo del escenario (char/string)
%   - type: Clasificación del escenario (char/string)
%   - inclusiones: Cell array de structs con campos x, y, radius
%
%   Coordenadas:
%   - x, y: Posición del centro de la inclusión (dominio normalizado)
%   - radius: Radio de la inclusión circular
%   - Dominio asumido: círculo unitario centrado en el origen
%
% Ejemplo:
%   scenarios = get_scenarios_definition();
%   escenario_1 = scenarios{1};
%   fprintf('Nombre: %s, Tipo: %s\n', escenario_1.name, escenario_1.type);
%
% Ver también: generar_imagen_conductividad, add_circular_inclusion

%% Inicializar cell array de escenarios
scenarios = cell(1, 10);

%% Escenario 1: Homogéneo (sin inclusiones)
scenarios{1} = struct();
scenarios{1}.name = 'homogeneo';
scenarios{1}.type = 'homogeneo';
scenarios{1}.inclusiones = {};

%% Escenario 2: Inclusión central pequeña
scenarios{2} = struct();
scenarios{2}.name = 'central_pequeno';
scenarios{2}.type = 'simple';
scenarios{2}.inclusiones = {
   struct('x', 0, 'y', 0, 'radius', 0.1)
};

%% Escenario 3: Inclusión central grande
scenarios{3} = struct();
scenarios{3}.name = 'central_grande';
scenarios{3}.type = 'simple';
scenarios{3}.inclusiones = {
   struct('x', 0, 'y', 0, 'radius', 0.2)
};

%% Escenario 4: Inclusión excéntrica
scenarios{4} = struct();
scenarios{4}.name = 'excentrico';
scenarios{4}.type = 'simple';
scenarios{4}.inclusiones = {
   struct('x', 0.3, 'y', -0.2, 'radius', 0.15)
};

%% Escenario 5: Dos inclusiones simétricas
scenarios{5} = struct();
scenarios{5}.name = 'dual_simetrico';
scenarios{5}.type = 'multiple';
scenarios{5}.inclusiones = {
   struct('x', 0.25, 'y', 0.25, 'radius', 0.1), ...
   struct('x', -0.25, 'y', -0.25, 'radius', 0.1)
};

%% Escenario 6: Dos inclusiones asimétricas
scenarios{6} = struct();
scenarios{6}.name = 'dual_asimetrico';
scenarios{6}.type = 'multiple';
scenarios{6}.inclusiones = {
   struct('x', 0.3, 'y', 0, 'radius', 0.12), ...
   struct('x', -0.2, 'y', 0.3, 'radius', 0.08)
};

%% Escenario 7: Triple configuración triangular
scenarios{7} = struct();
scenarios{7}.name = 'triple';
scenarios{7}.type = 'multiple';
scenarios{7}.inclusiones = {
   struct('x', 0, 'y', 0.3, 'radius', 0.08), ...
   struct('x', -0.26, 'y', -0.15, 'radius', 0.08), ...
   struct('x', 0.26, 'y', -0.15, 'radius', 0.08)
};

%% Escenario 8: Múltiples inclusiones en esquinas
scenarios{8} = struct();
scenarios{8}.name = 'multiple_esquinas';
scenarios{8}.type = 'multiple';
scenarios{8}.inclusiones = {
   struct('x', 0.4, 'y', 0.4, 'radius', 0.06), ...
   struct('x', -0.4, 'y', 0.4, 'radius', 0.06), ...
   struct('x', -0.4, 'y', -0.4, 'radius', 0.06), ...
   struct('x', 0.4, 'y', -0.4, 'radius', 0.06)
};

%% Escenario 9: Configuración anular (en cruz)
scenarios{9} = struct();
scenarios{9}.name = 'anular';
scenarios{9}.type = 'multiple';
scenarios{9}.inclusiones = {
   struct('x', 0.2, 'y', 0, 'radius', 0.05), ...
   struct('x', -0.2, 'y', 0, 'radius', 0.05), ...
   struct('x', 0, 'y', 0.2, 'radius', 0.05), ...
   struct('x', 0, 'y', -0.2, 'radius', 0.05)
};

%% Escenario 10: Configuración compleja (central + satélites)
scenarios{10} = struct();
scenarios{10}.name = 'complejo';
scenarios{10}.type = 'multiple';
scenarios{10}.inclusiones = {
   struct('x', 0, 'y', 0, 'radius', 0.05), ...        % Central pequeña
   struct('x', 0.25, 'y', 0.15, 'radius', 0.07), ...  % Satélite 1
   struct('x', -0.3, 'y', 0.1, 'radius', 0.09), ...   % Satélite 2
   struct('x', -0.15, 'y', -0.25, 'radius', 0.06), ... % Satélite 3
   struct('x', 0.2, 'y', -0.3, 'radius', 0.08)        % Satélite 4
};

%% Validación completa de la estructura de datos
for i = 1:length(scenarios)
   % Verificar que cada escenario tiene los campos obligatorios
   required_fields = {'name', 'type', 'inclusiones'};
   for j = 1:length(required_fields)
       if ~isfield(scenarios{i}, required_fields{j})
           error('get_scenarios_definition:MissingRequiredField', ...
               'El escenario %d no contiene el campo obligatorio: %s', i, required_fields{j});
       end
   end
   
   % Verificar tipos de datos de los campos principales
   if ~(ischar(scenarios{i}.name) || isstring(scenarios{i}.name))
       error('get_scenarios_definition:InvalidNameType', ...
           'El campo name del escenario %d debe ser char o string', i);
   end
   
   if ~(ischar(scenarios{i}.type) || isstring(scenarios{i}.type))
       error('get_scenarios_definition:InvalidTypeType', ...
           'El campo type del escenario %d debe ser char o string', i);
   end
   
   if ~iscell(scenarios{i}.inclusiones)
       error('get_scenarios_definition:InvalidInclusionsType', ...
           'El campo inclusiones del escenario %d debe ser un cell array', i);
   end
   
   % Validar tipo de escenario
   valid_types = {'homogeneo', 'simple', 'multiple'};
   if ~ismember(char(scenarios{i}.type), valid_types)
       error('get_scenarios_definition:InvalidScenarioType', ...
           'Tipo de escenario inválido en escenario %d: %s. Tipos válidos: %s', ...
           i, char(scenarios{i}.type), strjoin(valid_types, ', '));
   end
   
   % Validar cada inclusión dentro del escenario
   for k = 1:length(scenarios{i}.inclusiones)
       inclusion = scenarios{i}.inclusiones{k};
       
       % Verificar que es un struct
       if ~isstruct(inclusion)
           error('get_scenarios_definition:InvalidInclusionStructure', ...
               'La inclusión %d del escenario %d debe ser un struct', k, i);
       end
       
       % Verificar campos requeridos de la inclusión
       inclusion_fields = {'x', 'y', 'radius'};
       for m = 1:length(inclusion_fields)
           if ~isfield(inclusion, inclusion_fields{m})
               error('get_scenarios_definition:MissingInclusionField', ...
                   'La inclusión %d del escenario %d no contiene el campo: %s', ...
                   k, i, inclusion_fields{m});
           end
           
           % Verificar que es numérico y escalar
           field_value = inclusion.(inclusion_fields{m});
           if ~isnumeric(field_value) || ~isscalar(field_value) || ~isfinite(field_value)
               error('get_scenarios_definition:InvalidInclusionFieldValue', ...
                   'El campo %s de la inclusión %d del escenario %d debe ser un número escalar finito', ...
                   inclusion_fields{m}, k, i);
           end
       end
       
       % Validar rangos razonables
       if inclusion.radius <= 0
           error('get_scenarios_definition:InvalidRadius', ...
               'El radio de la inclusión %d del escenario %d debe ser positivo. Valor: %.3f', ...
               k, i, inclusion.radius);
       end
       
       if inclusion.radius > 0.5
           warning('get_scenarios_definition:LargeRadius', ...
               'El radio de la inclusión %d del escenario %d es muy grande: %.3f', ...
               k, i, inclusion.radius);
       end
       
       if abs(inclusion.x) > 0.8 || abs(inclusion.y) > 0.8
           warning('get_scenarios_definition:InclusionNearBoundary', ...
               'La inclusión %d del escenario %d está cerca del borde: (%.3f, %.3f)', ...
               k, i, inclusion.x, inclusion.y);
       end
   end
   
   % Validar consistencia entre tipo y número de inclusiones
   n_inclusiones = length(scenarios{i}.inclusiones);
   switch char(scenarios{i}.type)
       case 'homogeneo'
           if n_inclusiones ~= 0
               error('get_scenarios_definition:HomogeneousWithInclusions', ...
                   'El escenario homogéneo %d no debe tener inclusiones', i);
           end
       case 'simple'
           if n_inclusiones ~= 1
               warning('get_scenarios_definition:SimpleWithMultipleInclusions', ...
                   'El escenario simple %d tiene %d inclusiones (se esperaba 1)', i, n_inclusiones);
           end
       case 'multiple'
           if n_inclusiones < 2
               warning('get_scenarios_definition:MultipleWithFewInclusions', ...
                   'El escenario múltiple %d tiene solo %d inclusiones', i, n_inclusiones);
           end
   end
end

%% Añadir metadatos informativos a cada escenario
for i = 1:length(scenarios)
   scenarios{i}.scenario_id = i;
   scenarios{i}.num_inclusiones = length(scenarios{i}.inclusiones);
   
   % Calcular área total aproximada de inclusiones
   area_total = 0;
   for k = 1:length(scenarios{i}.inclusiones)
       area_total = area_total + pi * scenarios{i}.inclusiones{k}.radius^2;
   end
   scenarios{i}.area_total_inclusiones = area_total;
   scenarios{i}.fraccion_area_dominio = area_total / pi;  % Fracción del área total (círculo unitario)
end

end