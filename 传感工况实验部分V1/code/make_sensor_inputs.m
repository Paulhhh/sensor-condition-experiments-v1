function [base,variants,noise] = make_sensor_inputs(projectRoot,truth,cfg,rep,caseNames,gnssSensor)
% 生成 real_all 基线和观测变化。低频数组按采样历元排列。
% 星图像表每行依次为星号、测量 u/v、无噪声 u/v，坐标单位为像素。
if nargin < 6
    gnssSensor = build_fusion_sensor_data_real_gnss(truth,cfg,projectRoot);
end
indices = 1:cfg.fusion.gnss_step:numel(truth.t);
assert(cfg.fusion.gnss_step == cfg.fusion.star_step);
base.epochs = reshape(truth.t(indices),[],1);
assert(isequal(base.epochs,(0:600)'));
base.gyro = gnssSensor.imu.omega_m;
base.gyroR = gnssSensor.imu.R_gyro;
base.gnssR = gnssSensor.gnss.R;
base.gnssZ = gnssSensor.gnss.z(:,indices);
base.gnssValid = gnssSensor.gnss.valid(indices);
event = find(base.epochs >= 300 & base.epochs < 360);
base.rawGnss = compact_gnss(gnssSensor.gnss.raw(indices),base.epochs);
variants = struct();
noise = struct();
if any(strcmp(caseNames,'ar10'))
    [variants.ar10,noise] = correlated_gnss(base,gnssSensor,indices,event,cfg,rep);
end
clear gnssSensor

catalogFile = fullfile(projectRoot,'results','data','hipparcos_catalog_mag65.mat');
pairFile = fullfile(projectRoot,'results','data','hipparcos_pair_catalog_mag65.mat');
catalogData = load(catalogFile,'catalog_bright');
pairData = load(pairFile,'pair_catalog');
n = numel(base.epochs);
star.starQ = nan(4,n);
star.starValid = false(1,n);
star.starR = repmat(cfg.R_star,1,1,n);
star.inliers = zeros(1,n);
star.starReason = cell(1,n);
star.starImage = cell(1,n);
fprintf('Replicate %d: generating %d real star frames.\n',rep,n);
star = measure_stars(star,1:n,indices,truth,cfg,catalogData.catalog_bright, ...
    pairData.pair_catalog,rep,0.10);
fields = fieldnames(star);
for i = 1:numel(fields)
    base.(fields{i}) = star.(fields{i});
end
levels = [0.05 0.20 0.40];
names = {'pixel005','pixel020','pixel040'};
for i = 1:numel(levels)
    if any(strcmp(caseNames,names{i}))
        fprintf('Replicate %d: %.2f pixel event.\n',rep,levels(i));
        v = measure_stars(star,event,indices,truth,cfg,catalogData.catalog_bright, ...
            pairData.pair_catalog,rep,levels(i));
        for j = event(:)'
            a = star.starImage{j}; b = v.starImage{j};
            assert(isequal(a(:,[1 4 5]),b(:,[1 4 5])));
            residual = (a(:,2:3)-a(:,4:5))*(levels(i)/0.1)-(b(:,2:3)-b(:,4:5));
            assert(all(abs(residual(:)) < 1e-9),'Pixel noise draws did not match.');
        end
        variants.(names{i}) = v;
    end
end
end

function star = measure_stars(star,epochs,indices,truth,cfg,catalog,pairs,rep,sigma)
cameraConfig = cfg.star;
cameraConfig.camera.centroid_noise_std_pixel = sigma;
opts.min_stars_for_matching = 3;
opts.solve_attitude = true;
opts.save_intermediate = true;
opts.verify_with_true_id = true;
opts.meas_options.use_centroid_noise = true;
opts.match_options = struct('angle_tol_arcsec',60,'validation_tol_arcsec',120, ...
    'max_candidates_to_test',5000,'verbose',false);
for j = epochs(:)'
    k = indices(j);
    opts.t = truth.t(k);
    opts.epoch_index = k;
    opts.random_seed = 1000+j+(rep-1)*10000;
    out = star_sensor_full_logic(catalog,pairs,truth.r_i(k,:)',truth.q_bi(k,:)', ...
        truth.omega_b(k,:)',cfg,cameraConfig,opts);
    z = adapt_star_to_filter(out);
    star.starQ(:,j) = z.q_bi;
    star.starValid(j) = z.valid;
    star.starR(:,:,j) = cfg.R_star;
    if z.valid && ~isempty(z.R)
        star.starR(:,:,j) = z.R;
    end
    star.inliers(j) = out.info.num_inliers;
    star.starReason{j} = char(out.reason);
    star.starImage{j} = zeros(0,5);
    if isfield(out,'meas') && ~isempty(out.meas)
        m = out.meas;
        star.starImage{j} = [m.id(:),m.u_meas(:),m.v_meas(:),m.u_true(:),m.v_true(:)];
    end
end
end

function raw = compact_gnss(frames,epochs)
% 每行对应一颗可见卫星的一次观测。位置、速度使用地固坐标系。
counts = zeros(size(frames));
for j = 1:numel(frames)
    if ~isempty(frames{j})
        counts(j) = frames{j}.z.num_meas;
    end
end
n = sum(counts);
raw.epoch_s = zeros(n,1); raw.prn = zeros(n,1);
raw.sat_r_ecef_m = zeros(n,3); raw.sat_v_ecef_mps = zeros(n,3);
fields = {'rho_raw_m','rho_corr_m','rho_rate_raw_mps','rho_rate_corr_mps', ...
    'sat_clock_s','sat_clock_drift_sps','rho_noise_m','rate_noise_mps'};
source = {'rho_raw','rho_corr','rho_rate_raw','rho_rate_corr', ...
    'sat_clock_bias_s','sat_clock_drift_sps','noise','noise_rate'};
for i = 1:numel(fields)
    raw.(fields{i}) = zeros(n,1);
end
last = 0;
for j = 1:numel(frames)
    if counts(j)==0, continue; end
    z = frames{j}.z;
    rows = last + (1:counts(j));
    raw.epoch_s(rows) = epochs(j);
    raw.prn(rows) = str2double(extractAfter(string(z.sat_name),1));
    raw.sat_r_ecef_m(rows,:) = z.r_sat_ecef;
    raw.sat_v_ecef_mps(rows,:) = z.v_sat_ecef;
    for i = 1:numel(fields)
        raw.(fields{i})(rows) = z.(source{i});
    end
    last = rows(end);
end
assert(all(isfinite(raw.prn)));
end

function [v,noise] = correlated_gnss(base,sensor,indices,event,cfg,rep)
% 用正常可见历元的原始伪距噪声驱动相关序列；另用独立流补全不可见历元。
raw = base.rawGnss;
noise.prn = unique(raw.prn)';
n = numel(base.epochs); s = numel(noise.prn);
stream = RandStream('mt19937ar','Seed',730000+rep);
noise.white = 3*randn(stream,n,s);
[~,col] = ismember(raw.prn,noise.prn);
[~,row] = ismember(raw.epoch_s,base.epochs);
linear = sub2ind([n s],row,col);
assert(numel(unique(linear))==numel(row));
noise.white(linear) = raw.rho_noise_m;
a = exp(-0.1);
previous = 3*randn(stream,1,s);
noise.colored = zeros(n,s);
for j = 1:n
    previous = a*previous + sqrt(1-a^2)*noise.white(j,:);
    noise.colored(j,:) = previous;
end
v.gnssValid = base.gnssValid;
v.gnssZ = base.gnssZ;
v.rawGnss = raw;
active = ismember(row,event);
delta = noise.colored(linear(active))-raw.rho_noise_m(active);
v.rawGnss.rho_noise_m(active) = noise.colored(linear(active));
v.rawGnss.rho_raw_m(active)  = raw.rho_raw_m(active)+delta;
v.rawGnss.rho_corr_m(active) = raw.rho_corr_m(active)+delta;
opts = struct('max_iter',20,'tol_pos',1e-5,'tol_clock',1e-5,'verbose',false);
prev = find(base.gnssValid(1:event(1)-1),1,'last');
x0 = sensor.gnss.raw{indices(prev)}.nav.pos_sol.x;
for j = event(:)'
    k = indices(j);
    packet = sensor.gnss.raw{k};
    if isempty(packet) || numel(packet.meas)<4
        v.gnssZ(:,j) = nan; v.gnssValid(j) = false;
        continue
    end
    rows = find(row==j);
    m = packet.meas;
    assert(numel(rows)==numel(m));
    for i = 1:numel(m)
        m(i).rho_corr = v.rawGnss.rho_corr_m(rows(i));
    end
    position = gnss_position_ls(m,x0,opts);
    x0 = position.x;
    velocity = gnss_velocity_ls(m,position.r_user_ecef,struct('verbose',false));
    solved.nav = struct('valid',true,'r_ecef',position.r_user_ecef, ...
        'receiver_clock_bias_m',position.receiver_clock_bias_m);
    solved.vel = struct('valid',true,'v_ecef',velocity.v_user_ecef, ...
        'receiver_clock_drift_mps',velocity.receiver_clock_drift_mps);
    z = adapt_gnss_to_filter(solved,sensor.gnss.truth.C_ei(:,:,k),cfg.gnss.const.omega_e);
    v.gnssZ(:,j) = z.vec;
    v.gnssValid(j) = z.valid;
end
end
