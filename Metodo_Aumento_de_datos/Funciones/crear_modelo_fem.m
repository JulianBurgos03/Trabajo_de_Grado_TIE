function fmdl = crear_modelo_fem(n_electrodos)
% crear_modelo_fem - Crea un modelo FEM 2D de EIDORS para simulaciones EIT
%
% Sintaxis:
%   fmdl = crear_modelo_fem(n_electrodos)
%
% Entrada:
%   n_electrodos - Número de electrodos (debe ser 8 o 16)
%
% Salida:
%   fmdl - Estructura forward model de EIDORS configurada para EIT
%
% Descripción:
%   Esta función crea un modelo de elementos finitos 2D usando mk_common_model
%   de EIDORS y configura el patrón de estimulación adyacente. Es la función
%   base para todas las simulaciones EIT del proyecto de replicación del paper.
%
%   Configuración específica:
%   - 8 electrodos: usa modelo tipo 'a2c2'
%   - 16 electrodos: usa modelo tipo 'c2c2'
%   - Patrón adyacente para estimulación y medición
%   - Solver de primer orden para eficiencia computacional
%
% Ejemplo:
%   fmdl_8 = crear_modelo_fem(8);   % Modelo de 8 electrodos
%   fmdl_16 = crear_modelo_fem(16); % Modelo de 16 electrodos
%
% Ver también: mk_common_model, mk_stim_patterns, fwd_solve_1st_order

%% Validación de entrada
if nargin ~= 1
    error('crear_modelo_fem:WrongNumberOfInputs', ...
        'La función requiere exactamente un argumento de entrada');
end

if ~isnumeric(n_electrodos) || ~isscalar(n_electrodos) || n_electrodos ~= round(n_electrodos)
    error('crear_modelo_fem:InvalidInput', ...
        'n_electrodos debe ser un número entero');
end

if n_electrodos ~= 8 && n_electrodos ~= 16
    error('crear_modelo_fem:UnsupportedElectrodeCount', ...
        'n_electrodos debe ser 8 o 16. Valor recibido: %d', n_electrodos);
end

%% Verificar disponibilidad de EIDORS
if ~exist('mk_common_model', 'file')
    error('crear_modelo_fem:EIDORSNotAvailable', ...
        'EIDORS no está disponible. Asegúrese de que EIDORS esté instalado y en el path de MATLAB');
end

%% Crear modelo FEM usando mk_common_model
try
    if n_electrodos == 8
        modelo_temp = mk_common_model('a2c2', 8);
    elseif n_electrodos == 16
        modelo_temp = mk_common_model('c2c2', 16);
    end
catch ME
    error('crear_modelo_fem:ModelCreationFailed', ...
        'Error al crear el modelo FEM: %s', ME.message);
end

%% Extraer forward model
% mk_common_model devuelve un inverse model, extraemos el forward model
if isfield(modelo_temp, 'fwd_model')
    fmdl = modelo_temp.fwd_model;
else
    % Si no hay fwd_model, usar el modelo directamente (caso raro)
    fmdl = modelo_temp;
end

%% Configurar patrón de estimulación adyacente
try
    fmdl.stimulation = mk_stim_patterns(n_electrodos, 1, '{ad}', '{ad}');
catch ME
    error('crear_modelo_fem:StimulationPatternFailed', ...
        'Error al configurar el patrón de estimulación: %s', ME.message);
end

%% Configurar solver de primer orden
fmdl.solve = @fwd_solve_1st_order;
fmdl.jacobian = @jacobian_adjoint;

%% Validación del modelo creado
% Verificar que el modelo tiene la estructura correcta
if ~isfield(fmdl, 'electrode')
    error('crear_modelo_fem:InvalidModel', ...
        'El modelo creado no contiene información de electrodos');
end

if length(fmdl.electrode) ~= n_electrodos
    error('crear_modelo_fem:ElectrodeCountMismatch', ...
        'El modelo creado tiene %d electrodos, se esperaban %d', ...
        length(fmdl.electrode), n_electrodos);
end

if ~isfield(fmdl, 'stimulation') || isempty(fmdl.stimulation)
    error('crear_modelo_fem:NoStimulationPattern', ...
        'El modelo no tiene patrón de estimulación configurado');
end

% Verificar campos esenciales del modelo FEM
required_fields = {'nodes', 'elems', 'electrode', 'stimulation'};
for i = 1:length(required_fields)
    if ~isfield(fmdl, required_fields{i})
        error('crear_modelo_fem:MissingRequiredField', ...
            'El modelo no contiene el campo requerido: %s', required_fields{i});
    end
end

end