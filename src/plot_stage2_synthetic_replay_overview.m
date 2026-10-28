function figureFiles = plot_stage2_synthetic_replay_overview(inputArg, options)
% Plot synthetic replay wind/air estimates, air errors, and gate diagnostics.

if nargin < 1 || isempty(inputArg)
    inputArg = 'stage2_synthetic_online_replay_gamma5_20_results.mat';
end
if nargin < 2 || isempty(options)
    options = struct();
end

results = localLoadResults(inputArg);
if ~isfield(results, 'entries') || isempty(results.entries)
    error('Expected synthetic replay results with non-empty results.entries.');
end

prefix = char(localField(options, 'prefix', 'stage2_synthetic_online_replay_gamma5_20'));
windFile = [prefix '_wind_estimates_errors.png'];
airFile = [prefix '_air_estimates_errors.png'];
gateFile = [prefix '_gate_mode_lambda.png'];

D = localConcatEntries(results.entries);

f1 = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1300 820]);
windLabels = {'Vw_N (ft/s)', 'Vw_E (ft/s)'};
for ch = 1:2
    subplot(2, 2, ch);
    localPlotEstimateSet(D.t, D.windTrue(ch, :), D.windEkf(ch, :), ...
        D.windLstm(ch, :), D.windFuse(ch, :));
    grid on;
    ylabel(windLabels{ch});
    if ch == 1
        title('Synthetic Replay Wind Estimates');
        legend('true', 'EKF', 'LSTM aux', 'Fusion', 'Location', 'best');
    end

    subplot(2, 2, ch + 2);
    localPlotErrorSet(D.t, D.windEkf(ch, :) - D.windTrue(ch, :), ...
        D.windLstm(ch, :) - D.windTrue(ch, :), D.windFuse(ch, :) - D.windTrue(ch, :));
    grid on;
    ylabel([windLabels{ch} ' err']);
    xlabel('Time (s)');
    if ch == 1
        title('Synthetic Replay Wind Errors');
        legend('EKF', 'LSTM aux', 'Fusion', 'Location', 'best');
    end
end
saveas(f1, windFile);
close(f1);

f2 = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1300 920]);
airLabels = {'VT (ft/s)', 'alpha (deg)', 'beta (deg)'};
airTrue = [D.airTrue(1, :); rad2deg(D.airTrue(2:3, :))];
airEkf = [D.airEkf(1, :); rad2deg(D.airEkf(2:3, :))];
airLstm = [D.airLstm(1, :); rad2deg(D.airLstm(2:3, :))];
airFuse = [D.airFuse(1, :); rad2deg(D.airFuse(2:3, :))];
for ch = 1:3
    subplot(3, 2, 2 * ch - 1)
    localPlotEstimateSet(D.t, airTrue(ch, :), airEkf(ch, :), airLstm(ch, :), airFuse(ch, :));
    grid on;
    ylabel(airLabels{ch});
    if ch == 1
        title('Synthetic Replay Air Estimates');
        legend('true', 'EKF', 'LSTM aux', 'Fusion', 'Location', 'best');
    end
    if ch == 3
        xlabel('Time (s)');
    end

    subplot(3, 2, 2 * ch)
    localPlotErrorSet(D.t, airEkf(ch, :) - airTrue(ch, :), ...
        airLstm(ch, :) - airTrue(ch, :), airFuse(ch, :) - airTrue(ch, :));
    grid on;
    ylabel([airLabels{ch} ' err']);
    if ch == 1
        title('Synthetic Replay Air Errors');
        legend('EKF', 'LSTM aux', 'Fusion', 'Location', 'best');
    end
    if ch == 3
        xlabel('Time (s)');
    end
end
saveas(f2, airFile);
close(f2);

thresholds = localResolveThresholds(results);
f3 = figure('Visible', 'off', 'Color', 'w', 'Position', [120 120 1300 860]);
subplot(4, 1, 1)
stairs(D.t, D.mode, 'LineWidth', 1.0);
grid on;
ylim([-0.2 3.2]);
ylabel('mode');
title('Synthetic Replay Gate / Mode / Lambda');

subplot(4, 1, 2)
plot(D.t, D.lambda1, 'Color', [0.15 0.35 0.75], 'LineWidth', 1.0); hold on;
yline(thresholds.T1, '--', 'T1');
grid on;
ylabel('lambda1');

subplot(4, 1, 3)
plot(D.t, D.lambda2, 'Color', [0.82 0.22 0.16], 'LineWidth', 1.0); hold on;
yline(thresholds.T2, '--', 'T2');
grid on;
ylabel('lambda2');

subplot(4, 1, 4)
stairs(D.t, double(D.severeMask), 'Color', [0.70 0.10 0.10], 'LineWidth', 1.0); hold on;
stairs(D.t, double(D.substMask), 'Color', [0.10 0.45 0.20], 'LineWidth', 1.0);
grid on;
ylim([-0.1 1.1]);
ylabel('mask');
xlabel('Time (s)');
legend('severe', 'subst', 'Location', 'best');
saveas(f3, gateFile);
close(f3);

figureFiles = string({windFile; airFile; gateFile});
fprintf('saved figure: %s\n', windFile);
fprintf('saved figure: %s\n', airFile);
fprintf('saved figure: %s\n', gateFile);
end

function results = localLoadResults(inputArg)
if isstruct(inputArg)
    results = inputArg;
elseif ischar(inputArg) || isstring(inputArg)
    S = load(char(inputArg), 'results');
    if ~isfield(S, 'results')
        error('Missing results variable in %s.', char(inputArg));
    end
    results = S.results;
else
    error('Unsupported inputArg type.');
end
end

function D = localConcatEntries(entries)
gap = NaN;
D = struct();
fields = {'t', 'windTrue', 'windEkf', 'windLstm', 'windFuse', ...
    'airTrue', 'airEkf', 'airLstm', 'airFuse', 'mode', ...
    'lambda1', 'lambda2', 'severeMask', 'substMask'};
for i = 1:numel(fields)
    D.(fields{i}) = [];
end

tOffset = 0;
for i = 1:numel(entries)
    r = entries(i);
    %t = double(r.online.time(:)') + tOffset;
    t = double(r.online.time(:)');
    dt = localMedianDt(t);
    D.t = [D.t, t, gap]; %#ok<AGROW>
    D.windTrue = [D.windTrue, double(r.online.windTrue), nan(2, 1)]; %#ok<AGROW>
    D.windEkf = [D.windEkf, double(r.online.windEkf), nan(2, 1)]; %#ok<AGROW>
    D.windLstm = [D.windLstm, localAuxOrNan(r.replay, 'windLstmAux', 2, numel(t)), nan(2, 1)]; %#ok<AGROW>
    D.windFuse = [D.windFuse, double(r.online.windFuse), nan(2, 1)]; %#ok<AGROW>
    D.airTrue = [D.airTrue, double(r.online.airTrue), nan(3, 1)]; %#ok<AGROW>
    D.airEkf = [D.airEkf, double(r.online.airEkf), nan(3, 1)]; %#ok<AGROW>
    D.airLstm = [D.airLstm, localAuxOrNan(r.replay, 'airLstmAux', 3, numel(t)), nan(3, 1)]; %#ok<AGROW>
    D.airFuse = [D.airFuse, double(r.online.airFuse), nan(3, 1)]; %#ok<AGROW>
    D.mode = [D.mode, double(r.online.mode), gap]; %#ok<AGROW>
    D.lambda1 = [D.lambda1, double(r.online.lambda1), gap]; %#ok<AGROW>
    D.lambda2 = [D.lambda2, double(r.online.lambda2), gap]; %#ok<AGROW>
    severeMask = r.replayFault.subst_label ~= 0 | r.replayFault.anomaly_severity >= 3;
    substMask = r.replayFault.subst_label ~= 0;
    D.severeMask = [D.severeMask, logical(severeMask), false]; %#ok<AGROW>
    D.substMask = [D.substMask, logical(substMask), false]; %#ok<AGROW>
    tOffset = t(end) + max(dt, 0.01);
end
end

function dt = localMedianDt(t)
if numel(t) < 2
    dt = 0.01;
else
    dt = median(diff(t));
end
end

function M = localAuxOrNan(replay, field, rows, N)
if isstruct(replay) && isfield(replay, field) && ~isempty(replay.(field))
    M = double(replay.(field));
    M = M(:, 1:min(N, size(M, 2)));
    if size(M, 2) < N
        M(:, end + 1:N) = NaN;
    end
else
    M = nan(rows, N);
end
end

function localPlotEstimateSet(t, truth, ekf, lstmAux, fusion)
plot(t, truth, 'k', 'LineWidth', 1.2); hold on;
plot(t, ekf, '--', 'Color', [0.15 0.35 0.75], 'LineWidth', 1.0);
plot(t, lstmAux, ':', 'Color', [0.10 0.55 0.15], 'LineWidth', 1.1);
plot(t, fusion, '-', 'Color', [0.82 0.22 0.16], 'LineWidth', 1.0);
end

function localPlotErrorSet(t, ekfErr, lstmErr, fusionErr)
plot(t, ekfErr, '--', 'Color', [0.15 0.35 0.75], 'LineWidth', 1.0); hold on;
plot(t, lstmErr, ':', 'Color', [0.10 0.55 0.15], 'LineWidth', 1.1);
plot(t, fusionErr, '-', 'Color', [0.82 0.22 0.16], 'LineWidth', 1.0);
yline(0, ':k');
end

function thresholds = localResolveThresholds(results)
thresholds = struct('T1', 16.812, 'T2', 14.449, 'T1ch', 6.635, 'T2ch', 5.024);
if isfield(results, 'gateStats') && isfield(results.gateStats, 'thresholds')
    src = results.gateStats.thresholds;
elseif isfield(results.entries(1), 'gateStats') && isfield(results.entries(1).gateStats, 'thresholds')
    src = results.entries(1).gateStats.thresholds;
else
    return;
end
names = fieldnames(thresholds);
for i = 1:numel(names)
    key = names{i};
    if isstruct(src) && isfield(src, key)
        thresholds.(key) = double(src.(key));
    end
end
end

function value = localField(S, field, defaultValue)
if isstruct(S) && isfield(S, field)
    value = S.(field);
else
    value = defaultValue;
end
end
