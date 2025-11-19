function score = custom_fitness(alpha, vi1_meas, vi2_meas)
    vi1_ev = alpha * (vi1_meas(1:end-1) - vi1_meas(2:end)) + vi1_meas(2:end);
    min_len = min(length(vi1_meas), length(vi1_ev));
    Conjugada1 = zeros(2 * min_len, 1);
    Conjugada1(1:2:end) = vi1_meas(1:min_len);
    Conjugada1(2:2:end) = vi1_ev(1:min_len);

    if length(Conjugada1) < 128
        falta = 128 - length(Conjugada1);
        Conjugada1 = [Conjugada1; interp1(1:length(Conjugada1), Conjugada1, linspace(1, length(Conjugada1), falta), 'linear')'];
    end

    Measurements = reshape(Conjugada1(1:128), [16 8])';
    Dezpl = @(x, sh) circshift(Measurements(x,:), sh);
    C = @(d) interp1(1:16, d, linspace(1, 16, 16), 'linear');
    offset1 = -0.25E-4;
    Conjugada = offset1 + [ ...
        Measurements(1,:), C(Dezpl(2,-1)), Measurements(2,:), C(Dezpl(3,-1)), ...
        Measurements(3,:), C(Dezpl(4,-1)), Measurements(4,:), C(Dezpl(5,-1)), ...
        Measurements(5,:), C(Dezpl(6,-1)), Measurements(6,:), C(Dezpl(7,-1)), ...
        Measurements(7,:), C(Dezpl(8,-1)), Measurements(8,:), C(Dezpl(8, 1)) ...
    ]';

    len = min(length(Conjugada), length(vi2_meas));
    v1 = Conjugada(1:len);
    v2 = vi2_meas(1:len);

    ER = norm(v1 - v2) / norm(v2) * 100;
    MAE = mean(abs(v1 - v2));
    R = corr(v1, v2);

    score = 3 * ER^2 + 1000 * MAE + 10 * (1 - R);  % Score compuesto
end
