function run_sensor_experiments(reps, firstGnssCache)
% Run the four paired sensor experiments with the original 20-state ESKF.
% Each repetition creates complete, independently replayable observations.
if nargin < 1, reps = 1:5; end
if nargin < 2, firstGnssCache = ''; end
experimentDir = fileparts(fileparts(mfilename('fullpath')));
projectRoot = fullfile(fileparts(experimentDir), 'satellite_nav_sim');
dataDir = fullfile(experimentDir, 'data');
if ~isfolder(dataDir), mkdir(dataDir); end
addpath(genpath(fullfile(projectRoot,'sensors')), genpath(fullfile(projectRoot,'fusion')), ...
    fullfile(projectRoot,'config'),fullfile(projectRoot,'truth'),fullfile(projectRoot,'utils'));
nominal = config_fusion();
truth = generate_truth_all(nominal);
t = truth.t(:); r_true = truth.r_i; v_true = truth.v_i; q_true = truth.q_bi;
omega_true = truth.omega_b;
assert(numel(t)==60001 && abs(t(end)-600)<1e-10);
save(fullfile(dataDir,'truth.part.mat'),'t','r_true','v_true','q_true','omega_true','-v7');
movefile(fullfile(dataDir,'truth.part.mat'), fullfile(dataDir,'truth.mat'),'f');
for rep = reps
    cfg = nominal;
    offset = 10000*(rep-1);
    cfg.fusion.seed_init = nominal.fusion.seed_init + offset;
    cfg.imu.random_seed = nominal.imu.random_seed + offset;
    cfg.fusion.seed_real_gnss = nominal.fusion.seed_real_gnss + offset;
    names = ["baseline","star_sparse","star_burst","gnss_sparse","gnss_burst"];
    if rep==1
        names = [names,"pixel005","pixel020","pixel040","ar10","delay005","delay010","delay020"];
    else
        names = [names,"pixel040","ar10","delay010"];
    end
    prefix = fullfile(dataDir,sprintf('rep%02d_',rep));
    fprintf('Generating paired inputs, repetition %d\n',rep);
    if rep==1 && strlength(firstGnssCache)>0
        cache = load(firstGnssCache,'cachedGnss');
        [base,variants,noise] = make_sensor_inputs(projectRoot,truth,cfg,rep,names,cache.cachedGnss);
        clear cache
    else
        [base,variants,noise] = make_sensor_inputs(projectRoot,truth,cfg,rep,names);
    end
    idx = make_state_index();
    [X0,P0] = init_fusion_state(truth,cfg,idx);
    save([prefix 'base.part.mat'],'base','noise','cfg','rep','X0','P0','-v7');
    movefile([prefix 'base.part.mat'],[prefix 'base.mat'],'f');
    saved = struct();
    for name = names
        obs = condition_observations(base,variants,name);
        [out,checkpoint] = replay(truth,base,obs,cfg,idx,X0,P0,saved);
        if name=="baseline"
            saved = checkpoint;
            saved.baseline = out;
            verify_baseline(out,obs,rep);
        else
            assert(isequal(out.att(1:30000),saved.baseline.att(1:30000)));
            assert(isequal(out.dpos(1:30000,:),saved.baseline.dpos(1:30000,:)));
        end
        assert(all(isfinite([out.pos;out.vel;out.att])));
        assert(out.numStar==nnz(obs.starValid(2:end)) && out.numGnss==nnz(obs.gnssValid(2:end)));
        temp = [prefix char(name) '.part.mat'];
        save(temp,'name','rep','out','obs','-v7');
        movefile(temp,[prefix char(name) '.mat'],'f');
        fprintf('DONE %02d %s  GNSS=%d star=%d\n',rep,name,out.numGnss,out.numStar);
    end
    clear base variants noise saved
end
end

function obs = condition_observations(base,variants,name)
for field = ["epochs","starQ","starR","starValid","inliers","gnssZ","gnssValid"]
    obs.(field) = base.(field);
end
obs.arrival = obs.epochs;
if isfield(variants,name)
    changed = variants.(name);
    for field = string(fieldnames(changed)).'
        obs.(field) = changed.(field);
    end
elseif startsWith(name,"delay")
    delay = str2double(extractAfter(name,"delay"))/100;
    event = obs.epochs>=300 & obs.epochs<360;
    obs.arrival(event) = obs.epochs(event) + delay;
elseif endsWith(name,"sparse") || endsWith(name,"burst")
    if endsWith(name,"sparse"), deleted = 301:2:319; else, deleted = 306:315; end
    field = 'gnssValid';
    if startsWith(name,"star"), field = 'starValid'; end
    selected = ismember(obs.epochs,deleted);
    assert(nnz(selected)==10 && all(obs.(field)(selected)),'Expected ten originally valid observations.');
    obs.(field)(selected) = false;
end
end

function [out,checkpoint] = replay(truth,base,obs,cfg,idx,X0,P0,saved)
t = truth.t(:); n = numel(t);
% At 299.99 s every condition still has the same observations and state.
split = 30000;
starAt = zeros(n,1); gnssAt = zeros(n,1);
sv = find(obs.starValid); gv = find(obs.gnssValid);
starGrid = round(obs.arrival(sv)/cfg.sim.dt)+1;
gnssGrid = round(obs.epochs(gv)/cfg.sim.dt)+1;
assert(all(diff(starGrid)>0) && all(diff(gnssGrid)>0));
assert(max(abs(t(starGrid)-reshape(obs.arrival(sv),[],1)))<1e-9);
starAt(starGrid) = sv; gnssAt(gnssGrid) = gv;
isBase = isempty(fieldnames(saved));
if isBase
    X = X0; P = P0; first = 1;
    out = struct('pos',zeros(n,1),'vel',zeros(n,1),'att',zeros(n,1),'dpos',zeros(n,3), ...
        'numStar',0,'numGnss',0);
    out.dpos(1,:) = X.r.'-truth.r_i(1,:);
    out.pos(1) = norm(out.dpos(1,:)); out.vel(1) = norm(X.v.'-truth.v_i(1,:));
    out.att(1) = rad2deg(quat_angle_error(X.q_bi,truth.q_bi(1,:)))*3600;
else
    X = saved.X; P = saved.P; first = split;
    out = saved.baseline; out.numStar = saved.numStar; out.numGnss = saved.numGnss;
end
checkpoint = struct();
for k = first:n-1
    [X,P] = eskf_update_gyro(X,P,base.gyro(:,k),base.gyroR,idx);
    [X,P] = eskf_predict_full(X,P,t(k),t(k+1)-t(k),cfg,idx);
    next = k+1; g = gnssAt(next); s = starAt(next);
    if g>0
        [X,P] = eskf_update_gnss(X,P,obs.gnssZ(:,g),base.gnssR,idx);
        out.numGnss = out.numGnss+1;
    end
    if s>0
        [X,P] = eskf_update_star(X,P,obs.starQ(:,s),obs.starR(:,:,s),idx);
        out.numStar = out.numStar+1;
    end
    dp = X.r.' - truth.r_i(next,:);
    out.dpos(next,:) = dp; out.pos(next) = norm(dp);
    out.vel(next) = norm(X.v.'-truth.v_i(next,:));
    out.att(next) = rad2deg(quat_angle_error(X.q_bi,truth.q_bi(next,:)))*3600;
    if isBase && next==split
        checkpoint = struct('X',X,'P',P,'numStar',out.numStar,'numGnss',out.numGnss);
    end
end
end

function verify_baseline(out,obs,rep)
if rep~=1, return; end
assert(all(obs.starValid) && all(obs.gnssValid));
k = 101:100:numel(out.pos);
rmsPos = sqrt(mean(out.pos(k).^2)); rmsAtt = sqrt(mean(out.att(k).^2));
fprintf('Archived baseline comparison: %.9f m, %.9f arcsec\n',rmsPos,rmsAtt);
assert(abs(rmsPos-0.610387198)<1e-5 && abs(rmsAtt-9.127131)<0.001, ...
    'Baseline disagrees with archived real_all result.');
end
