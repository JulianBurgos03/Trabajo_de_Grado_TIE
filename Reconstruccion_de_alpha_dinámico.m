% ====================== main_hybrid_residual_v3.m ======================
% Híbrido Físico+NN (residuo escalado) con baseline físico fuerte,
% oversampling en zonas difíciles y CV por posiciones j.
close all; clear; clc; rng(42);

%% 0) Archivo y carga
phantom_file = 'C:\Users\juanp\OneDrive\Escritorio\TG_EIT\Datasets\DATA_ALPHA_Dynamic\plastic_2_45.txt';
D = read_eidors_phantom_txt(phantom_file);
fprintf('Archivo: %s\nMedidas: %d | Electrodos: %d | Estrategia: %s | f=%.1f kHz\n', ...
  phantom_file, numel(D.V), D.meta.electrodes, strtrim(D.meta.strategy), D.meta.freq_khz);

%% 1) Señal y estructura por bloques
v = abs(D.V ./ max(D.I, eps)); v = v(:);
E = D.meta.electrodes; M = numel(v); mps = round(M/E);
Y = reshape(v, mps, E);                       % columnas=bloques (1..E)
P = 1:2:E;  V = 2:2:E;                         % impares reales, pares virtuales
L1 = mod(V-2,E)+1; R1 = mod(V,E)+1;            % vecinos 1er anillo
L2 = mod(V-3,E)+1; R2 = mod(V+1,E)+1;          % vecinos 2º anillo

%% 2) Baseline físico fuerte (joint + gating + sesgo + post-ganancia)
BL = struct('perc',94,'lambda',1e-12,'mu',2e-2,'c0',3e-2, ...
            'gthr_base',0.18,'beta',0.35,'bounds1',[0 1],'bounds2',[-0.2 0.2]);
[YhatV_base, A1, A2, bj] = compute_baseline(Y, V, L1,R1, L2,R2, BL);

%% 3) Dataset para la NN (residuo escalado por rango local)
[X, T, JIDX, UIDX, Rloc, yTrue, yBase] = build_dataset(Y, YhatV_base, V, L1,R1, L2,R2);
% Oversampling de "difíciles" (grandes |residuo base|)
[Xtr, Ttr] = oversample_hard(X, T, abs(yTrue - yBase));

%% 4) 5-fold CV por posiciones j
K = 5;
folds = make_folds_by_j(mps, K);
fprintf('\n== CV por posiciones j (%d-fold) ==\n', K);
ERs = zeros(K,1); MAEs = zeros(K,1); Corrs = zeros(K,1);

for k=1:K
  valJ = folds{k};      % posiciones de validación
  isVal = ismember(JIDX, valJ);
  Xtr_k = Xtr(~isVal,:); Ttr_k = Ttr(~isVal,:);
  Xtr_k = Xtr(~isVal,:); Ttr_k = Ttr(~isVal,:);
  Xva_k = X(isVal,:);    Tva_k = T(isVal,:);

  net = train_small_regressor(Xtr_k, Ttr_k, Xva_k, Tva_k);

  % Inferencia en validación
  That = predict(net, Xva_k);
  idxV = find(isVal);
  yb_val = yBase(idxV); rloc_val = Rloc(idxV);
  yhat_val = yb_val + rloc_val .* That;

  ytrue_val = yTrue(idxV);
  [ERs(k), MAEs(k), Corrs(k)] = metrics_block(yhat_val, ytrue_val);
  fprintf(' fold %d/%d -> ER=%.4f, MAE=%.3f, Corr=%.4f\n',k,K,ERs(k),MAEs(k),Corrs(k));
end
fprintf(' CV mean +-std -> ER=%.4f±%.4f | MAE=%.3f±%.3f | Corr=%.4f±%.4f\n', ...
  mean(ERs),std(ERs), mean(MAEs),std(MAEs), mean(Corrs),std(Corrs));

%% 5) Entrenamiento final con TODO y predicción híbrida
netFinal = train_small_regressor(Xtr, Ttr, X, T);
Tpred = predict(netFinal, X);
YhatV_hyb = reshape(yBase + Rloc .* Tpred, mps, numel(V));

%% 6) Métricas globales y comparación
YtrueV = Y(:,V);
[ER_base,MAE_base,Corr_base] = metrics_block(YhatV_base(:), YtrueV(:));
[ER_hyb, MAE_hyb, Corr_hyb ] = metrics_block(YhatV_hyb(:), YtrueV(:));

Ttbl = table(["Baseline físico (joint+bias)";"Híbrido Físico+NN (residual)"], ...
             [ER_base;ER_hyb],[MAE_base;MAE_hyb],[Corr_base;Corr_hyb], ...
             'VariableNames',{'metodo','ER','MAE','Corr'});
fprintf('\n== Comparativa (bloques V, Real vs Predicho) ==\n'); disp(Ttbl);

save('res_8FEM8EV_hybrid_residual_v3.mat','Ttbl','YhatV_base','YhatV_hyb','A1','A2','bj');

%% 7) Plots
figure('Name','Real vs EV (baseline vs híbrido)','Color','w');
subplot(1,2,1); plot(YtrueV(:),'k'); hold on; plot(YhatV_base(:),'m'); grid on;
title('Baseline físico'); legend('Real','EV-base'); xlabel('j dentro de V (concat.)'); ylabel('señal');
subplot(1,2,2); plot(YtrueV(:),'k'); hold on; plot(YhatV_hyb(:),'m'); grid on;
title('Híbrido Físico+NN (residuo)'); legend('Real','EV-híbrido'); xlabel('j dentro de V (concat.)'); ylabel('señal');

figure('Name','Conjugadas 16 reales vs 8FEM+8EV (baseline vs híbrido)','Color','w');
subplot(1,2,1);
Ym = Y; Ym(:,V) = YhatV_base; plot(abs(Y(:)),'b'); hold on; plot(abs(Ym(:)),'r'); grid on;
title('Baseline físico'); xlabel('Índice (por bloque concat.)'); ylabel('|Voltaje|');
legend('16 reales','8FEM+8EV');
subplot(1,2,2);
Ym = Y; Ym(:,V) = YhatV_hyb; plot(abs(Y(:)),'b'); hold on; plot(abs(Ym(:)),'r'); grid on;
title('Híbrido Físico+NN'); xlabel('Índice (por bloque concat.)'); ylabel('|Voltaje|');
legend('16 reales','8FEM+8EV');

% ====================== Funciones ======================

function [YhatV, a1, a2, bj] = compute_baseline(Y, V, L1,R1, L2,R2, O)
% Joint 2-anillos IRLS + gating "suave", con sesgo por posición (bj)
% y post-ganancia por bloque.
  mps = size(Y,1); U = numel(V);
  % Inicialización per-pos (1er anillo)
  a1 = fit_alpha_perpos_IRLS(Y,V,L1,R1,struct('lambda',O.lambda,'perc',O.perc,'bounds',O.bounds1));
  a2 = zeros(mps,1);
  % Gating vector por posición (adaptativo)
  gthr = zeros(mps,1);
  for j=1:mps
    d1=[]; for u=1:U, d1=[d1; Y(j,L1(u))-Y(j,R1(u))]; end %#ok<AGROW>
    gthr(j) = O.gthr_base*prctile(abs(d1),95);
  end
  % IRLS conjunto
  for it=1:6
    for j=1:mps
      yV=[]; d1=[]; d2=[];
      for u=1:U
        yV=[yV; Y(j,V(u)) - Y(j,R1(u))]; %#ok<AGROW>
        d1=[d1;  Y(j,L1(u)) - Y(j,R1(u))]; %#ok<AGROW>
        d2=[d2;  Y(j,L2(u)) - Y(j,R2(u))]; %#ok<AGROW>
      end
      r  = yV - (a1(j)*d1 + a2(j)*d2);
      q  = prctile(abs(r), O.perc); q=max(q,eps);
      w  = 1 ./ (1 + (abs(r)/q).^2);
      g  = double(abs(d2) >= gthr(j));     % gating
      w2 = w .* (1 + O.beta*g);           % sube peso si hay d2 útil
      A = [ (w2.*d1)'*d1 + O.lambda + 0*O.mu, (w2.*d1)'*d2 ; ...
            (w2.*d2)'*d1,                   (w2.*d2)'*d2 + O.lambda + O.c0 ];
      b = [ (w2.*d1)'*yV ; (w2.*d2)'*yV ];
      sol = A \ b;
      a1(j) = min(max(sol(1), O.bounds1(1)), O.bounds1(2));
      a2(j) = min(max(sol(2), O.bounds2(1)), O.bounds2(2));
    end
    % Suavidad leve
    if O.mu>0
      L = (diff(speye(mps),1)'*diff(speye(mps),1));
      a1 = (speye(mps)+O.mu*L)\a1;  a2 = (speye(mps)+O.mu*L)\a2;
      a1 = min(max(a1,O.bounds1(1)),O.bounds1(2));
      a2 = min(max(a2,O.bounds2(1)),O.bounds2(2));
    end
  end
  % Predicción inicial
  YhatV = predictEV_alpha12(Y,V,L1,R1,L2,R2,a1,a2,0);
  % Sesgo por posición (mediana del residual por j)
  R = Y(:,V) - YhatV;
  bj = median(R,2);  YhatV = YhatV + bj;
  % Post-ganancia por bloque (robusta)
  YtrueV = Y(:,V);
  YhatV = post_gain_per_block(YhatV, YtrueV, 90);
end

function YhatV = predictEV_alpha12(Y, V, L1,R1, L2,R2, a1, a2, clipc)
  [mps,U] = size(Y(:,V)); YhatV = zeros(mps,U);
  for u=1:U
    L=Y(:,L1(u)); R=Y(:,R1(u)); d1=L-R;
    Lb=Y(:,L2(u)); Rb=Y(:,R2(u)); d2=Lb-Rb;
    yhat = R + a1.*d1 + a2.*d2;
    if clipc>0
      d = max(abs(d1),abs(d2));
      lo = min(min(L,R),min(Lb,Rb)) - clipc*d;
      hi = max(max(L,R),max(Lb,Rb)) + clipc*d;
      yhat = min(max(yhat,lo),hi);
    end
    YhatV(:,u) = yhat;
  end
end

function a1 = fit_alpha_perpos_IRLS(Y,V,L1,R1,O)
  mps = size(Y,1); U=numel(V); a1=0.5*ones(mps,1);
  for j=1:mps
    yV=[]; d=[]; for u=1:U, yV=[yV; Y(j,V(u))-Y(j,R1(u))]; d=[d; Y(j,L1(u))-Y(j,R1(u))]; end %#ok<AGROW>
    aj=0.5;
    for it=1:6
      r=yV-aj*d; q=prctile(abs(r),O.perc); q=max(q,eps);
      w=1./(1+(abs(r)/q).^2);
      aj=((w.*d)'*(w.*yV))/((w.*d)'*(w.*d)+O.lambda);
      aj=min(max(aj,O.bounds(1)),O.bounds(2));
    end
    a1(j)=aj;
  end
end

function Yhat = post_gain_per_block(Yhat, Ytrue, perc)
  [mps,U]=size(Yhat);
  for u=1:U
    yh=Yhat(:,u); yt=Ytrue(:,u); r=yt-yh;
    q=prctile(abs(r), perc); q=max(q,eps);
    w=1./(1+(abs(r)/q).^2);
    g=((w.*yh)'*(w.*yt))/((w.*yh)'*(w.*yh)+1e-12);
    Yhat(:,u)=g*yh;
  end
end

function [X,T,JIDX,UIDX,Rloc,yTrue,yBase] = build_dataset(Y, YhatV_base, V, L1,R1, L2,R2)
  [mps,U] = size(Y(:,V));
  N = mps*U;
  X   = zeros(N, 14);
  T   = zeros(N, 1);
  JIDX= zeros(N, 1);
  UIDX= zeros(N, 1);
  yTrue = zeros(N,1);  yBase = zeros(N,1);
  Rloc  = zeros(N,1);
  k=1;
  for u=1:U
    L = Y(:,L1(u)); R = Y(:,R1(u)); d1=L-R;
    Lb= Y(:,L2(u)); Rb=Y(:,R2(u)); d2=Lb-Rb;
    yb = YhatV_base(:,u);  yt = Y(:,V(u));
    rloc = (abs(d1)+abs(d2))+1e-6;
    for j=1:mps
      jn = j/mps;
      un = u/U;
      feat = [ yb(j), d1(j), d2(j), L(j), R(j), Lb(j), Rb(j), ...
               abs(d1(j)), abs(d2(j)), max([L(j),R(j),Lb(j),Rb(j)]), ...
               min([L(j),R(j),Lb(j),Rb(j)]), sin(2*pi*jn), cos(2*pi*jn), ...
               sin(2*pi*un)];  % 14
      X(k,:) = feat;
      T(k)   = (yt(j)-yb(j))/rloc(j);
      Rloc(k)= rloc(j);  yTrue(k)=yt(j); yBase(k)=yb(j);
      JIDX(k)= j; UIDX(k)= u;
      k=k+1;
    end
  end
  % z-score de features
  mu = mean(X,1); sg = std(X,[],1)+1e-9;
  X = (X - mu)./sg;
end

function [Xo, To] = oversample_hard(X, T, ebase)
% ESTRATEGIA: duplicar (x2) el 20% con |e|>p80 y triplicar el 10% con |e|>p90
  p80 = prctile(ebase,80); p90 = prctile(ebase,90);
  idx2 = find(ebase>p80 & ebase<=p90);
  idx3 = find(ebase>p90);
  Xo = [X; X(idx2,:); X(idx3,:); X(idx3,:)];  % +1x y +2x
  To = [T; T(idx2,:); T(idx3,:); T(idx3,:)];
end

function folds = make_folds_by_j(mps, K)
  js = 1:mps;
  cv = cvpartition(mps,'KFold',K);
  folds = cell(K,1);
  for k=1:K
    folds{k} = js(cv.test(k));
  end
end

function net = train_small_regressor(Xtr, Ttr, Xva, Tva)
  % Asegura tipos y formas correctas
  Xtr = single(Xtr);     Xva = single(Xva);
  Ttr = single(Ttr(:));  Tva = single(Tva(:));

  % Red MLP pequeña para regresión (residuo)
  layers = [
    featureInputLayer(size(Xtr,2),'Name','in')
    fullyConnectedLayer(64,'Name','fc1','WeightsInitializer','he', ...
                        'WeightL2Factor',1,'BiasL2Factor',0)
    reluLayer('Name','r1')
    dropoutLayer(0.1,'Name','do1')
    fullyConnectedLayer(32,'Name','fc2','WeightsInitializer','he', ...
                        'WeightL2Factor',1,'BiasL2Factor',0)
    reluLayer('Name','r2')
    fullyConnectedLayer(1,'Name','out')
    regressionLayer('Name','reg')];

  opts = trainingOptions('adam', ...
    'InitialLearnRate',2e-3, ...
    'MiniBatchSize',128, ...
    'MaxEpochs',250, ...
    'Shuffle','every-epoch', ...
    'L2Regularization',1e-4, ...
    'ValidationData',{Xva,Tva}, ...
    'ValidationFrequency',30, ...
    'Verbose',false);
    % Si tu MATLAB no soporta ValidationPatience, simplemente quítalo.

  net = trainNetwork(Xtr, Ttr, layers, opts);
end


function [ER,MAE,Corr] = metrics_block(yhat, ytrue)
  ER  = norm(yhat(:)-ytrue(:)) / (norm(ytrue(:))+eps);
  MAE = mean(abs(yhat(:)-ytrue(:)));
  Corr= corr(real(yhat(:)), real(ytrue(:)), 'Rows','complete');
end

% -------- lector EIDORS --------
function D = read_eidors_phantom_txt(fp)
  fid=fopen(fp,'r'); assert(fid>0,'No puedo abrir: %s',fp);
  C=textscan(fid,'%s','Delimiter','\n','Whitespace',''); fclose(fid);
  L=C{1}; header=L(startsWith(strtrim(L),'#'));
  idxStart=find(startsWith(strtrim(L),'#Start'),1,'first'); assert(~isempty(idxStart));
  body=L(idxStart+1:end); body=body(~cellfun(@isempty,body)); body=body(~startsWith(strtrim(body),'#'));
  meta.length=first_int(header,'#Length');
  meta.electrodes=first_int(header,'#Electrodes');
  meta.freq_khz=first_num(header,'#Operating_Frequency');
  meta.strategy=after_comma(header,'#Strategy');
  M=numel(body); Vr=zeros(M,1); Vi=zeros(M,1); I=zeros(M,1);
  for k=1:M, vals=sscanf(body{k},'%f %f %f'); Vr(k)=vals(1); Vi(k)=vals(2); I(k)=vals(3); end
  if ~isempty(meta.length)&&numel(Vr)>meta.length, Vr=Vr(1:meta.length); Vi=Vi(1:meta.length); I=I(1:meta.length); end
  D.V=Vr+1i*Vi; D.I=I; D.meta=meta;
end
function x = first_int(H,key)
  m=find(startsWith(strtrim(H),key),1,'first'); if isempty(m), x=[]; else, a=regexp(H{m},'[-+]?\d+','match'); x=str2double(a{1}); end
end
function x = first_num(H,key)
  m=find(startsWith(strtrim(H),key),1,'first'); if isempty(m), x=[]; else, a=regexp(H{m},'[-+]?\d+(\.\d+)?([eE][-+]?\d+)?','match'); x=str2double(a{1}); end
end
function s = after_comma(H,key)
  m=find(startsWith(strtrim(H),key),1,'first'); if isempty(m), s=''; else, p=split(H{m},','); s=strtrim(strjoin(p(2:end),',')); end
end
% =================== end file ===================
%% ====================== RECONSTRUCCIÓN DE IMAGEN CON EIDORS (CORREGIDO) ======================
% Continuación después de obtener Y, YhatV_base, YhatV_hyb

%% 8) Configuración del modelo EIDORS ajustado al dataset real
n_elec = 16;
fprintf('\n== Configurando modelo EIDORS ==\n');

% Modelo base + estímulos Adyacentes (EIDORS ya da 13 mediciones por estímulo)
imdl = mk_common_model('c2c2', n_elec);
fmdl = imdl.fwd_model;
stim = mk_stim_patterns(n_elec, 1, '{ad}', '{ad}', {}, 1);
fmdl.stimulation = stim;
imdl.fwd_model = fmdl;

% Comprobación: medidas esperadas por EIDORS
n_stim = length(fmdl.stimulation);
mps_eidors = size(fmdl.stimulation(1).meas_pattern,1); % debería ser 13
n_meas_expected = n_stim * mps_eidors;
fprintf('Patrones de inyección: %d | Medidas por patrón: %d | Total esperado: %d\n', ...
        n_stim, mps_eidors, n_meas_expected);

% Tu matriz Y ya está como [mps x 16] con mps=13. Reordena a columna:
v_real_16 = Y(:);

% Ajuste de longitud si hiciera falta (por seguridad)
if numel(v_real_16) ~= n_meas_expected
    fprintf('Ajustando datos reales de %d a %d medidas\n', numel(v_real_16), n_meas_expected);
    if numel(v_real_16) > n_meas_expected
        v_real_16 = v_real_16(1:n_meas_expected);
    else
        v_real_16 = [v_real_16; repmat(mean(v_real_16), n_meas_expected - numel(v_real_16), 1)];
    end
end

% Conjugadas 8FEM+8EV (baseline e híbrido) en vector columna
Y_conj_base = Y;  Y_conj_base(:,V)  = YhatV_base;
Y_conj_hyb  = Y;  Y_conj_hyb(:,V)   = YhatV_hyb;

v_conj_base = Y_conj_base(:);
v_conj_hyb  = Y_conj_hyb(:);

% Ajustes defensivos (si hiciera falta)
if numel(v_conj_base) ~= n_meas_expected
    if numel(v_conj_base) > n_meas_expected
        v_conj_base = v_conj_base(1:n_meas_expected);
    else
        v_conj_base = [v_conj_base; repmat(mean(v_conj_base), n_meas_expected - numel(v_conj_base), 1)];
    end
end
if numel(v_conj_hyb) ~= n_meas_expected
    if numel(v_conj_hyb) > n_meas_expected
        v_conj_hyb = v_conj_hyb(1:n_meas_expected);
    else
        v_conj_hyb = [v_conj_hyb; repmat(mean(v_conj_hyb), n_meas_expected - numel(v_conj_hyb), 1)];
    end
end

% 9) Referencia de diferencia: usa vh homogéneo del MISMO fmdl
vh = fwd_solve(mk_image(fmdl, 1.0));
v_ref = vh.meas;

% (Opcional) Normaliza magnitud si tus datos ya son |V|:
% v_real_16  = abs(v_real_16);
% v_conj_base= abs(v_conj_base);
% v_conj_hyb = abs(v_conj_hyb);
% v_ref      = abs(v_ref);  % Normalmente NO se hace esto en difference
fprintf('Dimensiones finales: v_ref=%d, v_real=%d, v_base=%d, v_hyb=%d\n', ...
    length(v_ref), length(v_real_16), length(v_conj_base), length(v_conj_hyb));

%% 10) Algoritmos de reconstrucción
fprintf('\n== Iniciando reconstrucciones (Laplace, Tikhonov, TV) ==\n');

% 10.1 Laplace
fprintf('Ejecutando Laplace...\n');
imdl_laplace = imdl;
imdl_laplace.solve = @inv_solve_diff_GN_one_step;
imdl_laplace.RtR_prior = @prior_laplace;
imdl_laplace.hyperparameter.value = 0.05;
try
    img_laplace_real = inv_solve(imdl_laplace, v_ref, v_real_16);
    img_laplace_base = inv_solve(imdl_laplace, v_ref, v_conj_base);
    img_laplace_hyb  = inv_solve(imdl_laplace, v_ref, v_conj_hyb);
    laplace_ok = true; fprintf('  ✓ Laplace completado\n');
catch ME
    laplace_ok = false; fprintf('  ✗ Error en Laplace: %s\n', ME.message);
end

% 10.2 Tikhonov
fprintf('Ejecutando Tikhonov...\n');
imdl_tik = imdl;
imdl_tik.solve = @inv_solve_diff_GN_one_step;
imdl_tik.RtR_prior = @prior_tikhonov;
imdl_tik.hyperparameter.value = 0.08;
try
    img_tik_real = inv_solve(imdl_tik, v_ref, v_real_16);
    img_tik_base = inv_solve(imdl_tik, v_ref, v_conj_base);
    img_tik_hyb  = inv_solve(imdl_tik, v_ref, v_conj_hyb);
    tik_ok = true; fprintf('  ✓ Tikhonov completado\n');
catch ME
    tik_ok = false; fprintf('  ✗ Error en Tikhonov: %s\n', ME.message);
end

% 10.3 Total Variation
fprintf('Ejecutando Total Variation...\n');
imdl_tv = imdl;
imdl_tv.solve = @inv_solve_TV_pdipm;
imdl_tv.inv_solve_TV_pdipm.alpha = 5e-3;  % peso TV
imdl_tv.inv_solve_TV_pdipm.beta  = 1e-4;  % L2 leve
try
    img_tv_real = inv_solve(imdl_tv, v_ref, v_real_16);
    img_tv_base = inv_solve(imdl_tv, v_ref, v_conj_base);
    img_tv_hyb  = inv_solve(imdl_tv, v_ref, v_conj_hyb);
    tv_ok = true; fprintf('  ✓ Total Variation completado\n');
catch ME
    tv_ok = false; fprintf('  ✗ Error en Total Variation: %s\n', ME.message);
end

%% 11) Cálculo de métricas ER, MAE y CC (usando 16 Real como referencia)

fprintf('\n== Métricas de Reconstrucción (ER, MAE, CC) ==\n');
fprintf('Referencia: Reconstrucción con 16 electrodos reales\n\n');

% Tabla de resultados
metodos = {};
ER_base_vals = []; MAE_base_vals = []; CC_base_vals = [];
ER_hyb_vals = [];  MAE_hyb_vals = [];  CC_hyb_vals = [];

if laplace_ok
    % Base vs Real
    [er_lb, mae_lb, cc_lb] = metrics_reconstruction(img_laplace_base.elem_data, img_laplace_real.elem_data);
    % Híbrido vs Real
    [er_lh, mae_lh, cc_lh] = metrics_reconstruction(img_laplace_hyb.elem_data, img_laplace_real.elem_data);
    
    metodos = [metodos; 'Laplace'];
    ER_base_vals = [ER_base_vals; er_lb];   MAE_base_vals = [MAE_base_vals; mae_lb];   CC_base_vals = [CC_base_vals; cc_lb];
    ER_hyb_vals  = [ER_hyb_vals; er_lh];    MAE_hyb_vals  = [MAE_hyb_vals; mae_lh];    CC_hyb_vals  = [CC_hyb_vals; cc_lh];
    
    fprintf('Laplace:\n');
    fprintf('  8FEM+8EV Base:    ER=%.4f | MAE=%.4f | CC=%.4f\n', er_lb, mae_lb, cc_lb);
    fprintf('  8FEM+8EV Híbrido: ER=%.4f | MAE=%.4f | CC=%.4f\n\n', er_lh, mae_lh, cc_lh);
end

if tik_ok
    [er_tb, mae_tb, cc_tb] = metrics_reconstruction(img_tik_base.elem_data, img_tik_real.elem_data);
    [er_th, mae_th, cc_th] = metrics_reconstruction(img_tik_hyb.elem_data, img_tik_real.elem_data);
    
    metodos = [metodos; 'Tikhonov'];
    ER_base_vals = [ER_base_vals; er_tb];   MAE_base_vals = [MAE_base_vals; mae_tb];   CC_base_vals = [CC_base_vals; cc_tb];
    ER_hyb_vals  = [ER_hyb_vals; er_th];    MAE_hyb_vals  = [MAE_hyb_vals; mae_th];    CC_hyb_vals  = [CC_hyb_vals; cc_th];
    
    fprintf('Tikhonov:\n');
    fprintf('  8FEM+8EV Base:    ER=%.4f | MAE=%.4f | CC=%.4f\n', er_tb, mae_tb, cc_tb);
    fprintf('  8FEM+8EV Híbrido: ER=%.4f | MAE=%.4f | CC=%.4f\n\n', er_th, mae_th, cc_th);
end

if tv_ok
    [er_vb, mae_vb, cc_vb] = metrics_reconstruction(img_tv_base.elem_data, img_tv_real.elem_data);
    [er_vh, mae_vh, cc_vh] = metrics_reconstruction(img_tv_hyb.elem_data, img_tv_real.elem_data);
    
    metodos = [metodos; 'Total Variation'];
    ER_base_vals = [ER_base_vals; er_vb];   MAE_base_vals = [MAE_base_vals; mae_vb];   CC_base_vals = [CC_base_vals; cc_vb];
    ER_hyb_vals  = [ER_hyb_vals; er_vh];    MAE_hyb_vals  = [MAE_hyb_vals; mae_vh];    CC_hyb_vals  = [CC_hyb_vals; cc_vh];
    
    fprintf('Total Variation:\n');
    fprintf('  8FEM+8EV Base:    ER=%.4f | MAE=%.4f | CC=%.4f\n', er_vb, mae_vb, cc_vb);
    fprintf('  8FEM+8EV Híbrido: ER=%.4f | MAE=%.4f | CC=%.4f\n\n', er_vh, mae_vh, cc_vh);
end

% Tabla resumen
if ~isempty(metodos)
    Tbl_recon = table(metodos, ...
                      ER_base_vals, MAE_base_vals, CC_base_vals, ...
                      ER_hyb_vals, MAE_hyb_vals, CC_hyb_vals, ...
                      'VariableNames', {'Metodo', 'ER_Base', 'MAE_Base', 'CC_Base', ...
                                        'ER_Hibrido', 'MAE_Hibrido', 'CC_Hibrido'});
    fprintf('== Tabla Resumen de Métricas de Reconstrucción ==\n');
    disp(Tbl_recon);
end

%% 12) Visualización comparativa
if laplace_ok
    figure('Name','Reconstrucción Laplace: Comparativa','Color','w','Position',[50 50 1400 400]);
    subplot(1,3,1); show_fem(img_laplace_real); title('Laplace - 16 Real'); axis equal tight; colorbar;
    subplot(1,3,2); show_fem(img_laplace_base); title('Laplace - 8FEM+8EV Base'); axis equal tight; colorbar;
    subplot(1,3,3); show_fem(img_laplace_hyb); title('Laplace - 8FEM+8EV Híbrido'); axis equal tight; colorbar;
end

if tik_ok
    figure('Name','Reconstrucción Tikhonov: Comparativa','Color','w','Position',[50 550 1400 400]);
    subplot(1,3,1); show_fem(img_tik_real); title('Tikhonov - 16 Real'); axis equal tight; colorbar;
    subplot(1,3,2); show_fem(img_tik_base); title('Tikhonov - 8FEM+8EV Base'); axis equal tight; colorbar;
    subplot(1,3,3); show_fem(img_tik_hyb); title('Tikhonov - 8FEM+8EV Híbrido'); axis equal tight; colorbar;
end

if tv_ok
    figure('Name','Reconstrucción Total Variation: Comparativa','Color','w','Position',[1500 50 1400 400]);
    subplot(1,3,1); show_fem(img_tv_real); title('TV - 16 Real'); axis equal tight; colorbar;
    subplot(1,3,2); show_fem(img_tv_base); title('TV - 8FEM+8EV Base'); axis equal tight; colorbar;
    subplot(1,3,3); show_fem(img_tv_hyb); title('TV - 8FEM+8EV Híbrido'); axis equal tight; colorbar;
end

%% 13) Comparación de diferencias (visualizar errores)
if laplace_ok && tik_ok && tv_ok
    figure('Name','Diferencias: Híbrido vs Base (por método)','Color','w','Position',[1500 550 1400 400]);
    
    img_diff_laplace = img_laplace_base;
    img_diff_laplace.elem_data = img_laplace_hyb.elem_data - img_laplace_base.elem_data;
    
    img_diff_tik = img_tik_base;
    img_diff_tik.elem_data = img_tik_hyb.elem_data - img_tik_base.elem_data;
    
    img_diff_tv = img_tv_base;
    img_diff_tv.elem_data = img_tv_hyb.elem_data - img_tv_base.elem_data;
    
    subplot(1,3,1); show_fem(img_diff_laplace); title('Δ Laplace (Híb-Base)'); axis equal tight; colorbar;
    subplot(1,3,2); show_fem(img_diff_tik); title('Δ Tikhonov (Híb-Base)'); axis equal tight; colorbar;
    subplot(1,3,3); show_fem(img_diff_tv); title('Δ TV (Híb-Base)'); axis equal tight; colorbar;
end

%% 14) Guardar resultados
vars_to_save = {'imdl', 'fmdl', 'v_ref', 'v_real_16', 'v_conj_base', 'v_conj_hyb'};
if laplace_ok
    vars_to_save = [vars_to_save, {'img_laplace_real', 'img_laplace_base', 'img_laplace_hyb'}];
end
if tik_ok
    vars_to_save = [vars_to_save, {'img_tik_real', 'img_tik_base', 'img_tik_hyb'}];
end
if tv_ok
    vars_to_save = [vars_to_save, {'img_tv_real', 'img_tv_base', 'img_tv_hyb'}];
end
if exist('Tbl_recon', 'var')
    vars_to_save = [vars_to_save, {'Tbl_recon'}];
end

save('res_reconstruction_eidors.mat', vars_to_save{:});
fprintf('\n✓ Reconstrucciones completadas y guardadas.\n');

%% ============ Función auxiliar para métricas de reconstrucción ============
function [ER, MAE, CC] = metrics_reconstruction(img_test, img_ref)
    % Calcula métricas entre dos reconstrucciones
    % img_test: reconstrucción a evaluar
    % img_ref: reconstrucción de referencia (16 electrodos reales)
    
    img_test = img_test(:);
    img_ref = img_ref(:);
    
    % Error Relativo
    ER = norm(img_test - img_ref) / (norm(img_ref) + eps);
    
    % Mean Absolute Error
    MAE = mean(abs(img_test - img_ref));
    
    % Correlación
    CC = corr(real(img_test), real(img_ref), 'Rows', 'complete');
end