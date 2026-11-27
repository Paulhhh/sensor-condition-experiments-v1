function summarize_sensor_results(figureIds)
% 读取 44 次滤波与观测，生成四张主图、分段指标及同种子配对统计。
root = fileparts(fileparts(mfilename('fullpath')));
data = fullfile(root,'data');
dest = fullfile(root,'results');
if ~isfolder(dest), mkdir(dest); end
plotFile=fullfile(dest,'figure_inputs.mat');
if nargin==1 && ~isempty(figureIds)
    assert(all(ismember(figureIds,1:4)),'Figure IDs must be 1 through 4.');
    input=load(plotFile,'t','plotData','T');
    make_plots(dest,input.t,input.plotData,input.T,figureIds);
    return
end
truth = load(fullfile(data,'truth.mat'),'t','r_true','q_true','v_true');
t = truth.t(:);
rows = cell(0,12);
plotData = struct();
names = ["baseline","star_sparse","star_burst","gnss_sparse","gnss_burst", ...
    "pixel005","pixel020","pixel040","ar10","delay005","delay010","delay020"];
labels = ["正常基线","星敏分散缺测","星敏连续缺测","北斗分散缺测","北斗连续缺测", ...
    "星点噪声 0.05 像素","星点噪声 0.20 像素","星点噪声 0.40 像素","伪距时间相关噪声", ...
    "星敏延迟 0.05 s","星敏延迟 0.10 s","星敏延迟 0.20 s"];
for rep = 1:5
    if rep==1, active = names; else, active=names([1:5,8:9,11]); end
    prefix = fullfile(data,sprintf('rep%02d_',rep));
    B = load([prefix 'base.mat'],'noise');
    common = true(601,1);
    pixelData = struct();
    for id = active
        item = load([prefix char(id) '.mat'],'out','obs');
        o = item.obs; f = item.out;
        if id=="baseline", ref=o; end
        check_observations(id,o,ref);
        e = o.epochs(:); validS = logical(o.starValid(:)); validG = logical(o.gnssValid(:));
        arrivals = o.arrival(:);
        ix = round(e*100)+1;
        assert(max(abs(t(ix)-e))<1e-8);
        event = e>=300 & e<360;
        follow = t>=300 & t<420;
        lowFollow = e>=300 & e<420;
        q = o.starQ.';
        rawAtt = attitude_angle(q,truth.q_true(ix,:));
        arrivalAtt = attitude_angle(q,truth.q_true(round(arrivals*100)+1,:));
        rawXYZ = o.gnssZ(1:3,:).' - truth.r_true(ix,:);
        rawPos = vecnorm(rawXYZ,2,2);
        rawVel = vecnorm(o.gnssZ(4:6,:).'-truth.v_true(ix,:),2,2);
        win = ["before","event","after","drop","recovery","full_100Hz","full_1Hz"];
        lo = [240,300,360,301,320,1,1]; high = [300,360,420,319,340,600,600];
        row0 = size(rows,1);
        for j=1:numel(win)
            if j<=3
                k=t>=lo(j) & t<high(j); ek=e>=lo(j) & e<high(j);
            else
                k=t>=lo(j) & t<=high(j); ek=e>=lo(j) & e<=high(j);
            end
            if j==7
                indices=ix(ek);
            else
                indices=find(k);
            end
            if (j==4 || j==5) && ~(id=="baseline" || endsWith(id,"burst") || endsWith(id,"sparse")), continue; end
            nStar=nnz(ek&validS); nGnss=nnz(ek&validG);
            meta={rep,id,labels(names==id),win(j),nStar,nGnss};
            rows(end+1,:) = metric_row(meta,"fused_att","arcsec",f.att(indices));
            rows(end+1,:) = metric_row(meta,"fused_pos","m",f.pos(indices));
            rows(end+1,:) = metric_row(meta,"fused_vel","m/s",f.vel(indices));
            if j<=3 || j==7
                rows(end+1,:) = metric_row(meta,"star_sample_att","arcsec",rawAtt(ek&validS));
                rows(end+1,:) = metric_row(meta,"star_arrival_att","arcsec",arrivalAtt(ek&validS));
                rows(end+1,:) = metric_row(meta,"gnss_pos","m",rawPos(ek&validG));
                rows(end+1,:) = metric_row(meta,"gnss_vel","m/s",rawVel(ek&validG));
            end
            if j==2
                rows(end+1,:) = metric_row(meta,"star_valid","%",100*double(validS(ek)));
                rows(end+1,:) = metric_row(meta,"inlier_count_valid","stars",o.inliers(ek&validS));
                if startsWith(id,'pixel') || id=="baseline"
                    common = common & validS;
                    pixelData.(id)=struct('angle',rawAtt,'meta',{meta});
                end
                if id=="ar10" || id=="baseline"
                    for axis=1:3
                        letter=extractBetween("XYZ",axis,axis);
                        rows(end+1,:) = metric_row(meta,"gnss_"+letter,"m",rawXYZ(ek&validG,axis));
                        rows(end+1,:) = metric_row(meta,"fused_"+letter,"m",f.dpos(indices,axis));
                        raw1Hz=rawXYZ(ek,axis); fused1Hz=f.dpos(ix(ek),axis);
                        raw1Hz(~validG(ek))=NaN; fused1Hz(~validG(ek))=NaN;
                        rows(end+1,:)=metric_row(meta,"gnss_"+letter+"_same_sign_s","s",longest_same_sign(raw1Hz));
                        rows(end+1,:)=metric_row(meta,"fused_"+letter+"_same_sign_s","s",longest_same_sign(fused1Hz));
                    end
                end
            end
        end
        if id=="baseline" || id=="ar10"
            if id=="baseline", input=B.noise.white; else, input=B.noise.colored; end
            [acf,sd]=noise_stats(input,30);
            meta={rep,id,labels(names==id),"noise_0_600",nnz(validS),nnz(validG)};
            rows(end+1,:) = metric_row(meta,"rho_noise","m",input(:));
            rows(end+1,:) = metric_row(meta,"rho_std","m",sd);
            rows(end+1,:) = metric_row(meta,"rho_acf_lag1","1",acf(2));
            if rep==1, plotData.(id).acf=acf; end
        end
        if rep==1
            zoom=t>=295 & t<=340;
            P=struct('att',f.att(zoom),'vel',f.vel(zoom),'pos',f.pos(zoom));
            plotData.(id).outage=P;
            plotData.(id).followupAtt=f.att(follow);
            if id=="baseline" || id=="ar10"
                source=rawXYZ(lowFollow,:); source(~validG(lowFollow),:)=NaN;
                plotData.(id).rawXYZ=source;
                plotData.(id).fusedXYZ=f.dpos(follow,:);
            end
        end
        assert(size(rows,1)>row0);
        clear item o f
    end
    if rep==1, commonName="star_common4_att"; else, commonName="star_common2_att"; end
    for id = string(fieldnames(pixelData)).'
        dataForCase=pixelData.(id);
        rows(end+1,:)=metric_row(dataForCase.meta,commonName,"arcsec",dataForCase.angle(event&common));
    end
end
headers={'rep','case_id','case_name','window','valid_star','valid_gnss','metric','unit', ...
    'samples','mean','rms','peak_abs'};
T=cell2table(rows,'VariableNames',headers);
for key=["case_id","case_name","window","metric","unit"], T.(key)=string(T.(key)); end
writetable(T,fullfile(dest,'metrics.csv'),'Encoding','UTF-8');
write_pairs(T,dest);
% 缓存制图所需的数值，允许在单独的 MATLAB 进程中逐张绘图。
save(plotFile,'t','plotData','T','-v7');
if nargin==0
    make_plots(dest,t,plotData,T,1:4);
end
end

function check_observations(id,o,b)
e=b.epochs(:);
assert(isequal(o.epochs,b.epochs));
event=e>=300 & e<360;
starMask=logical(b.starValid(:)); gnssMask=logical(b.gnssValid(:));
removed=false(size(e));
if endsWith(id,"sparse"), removed=ismember(e,301:2:319); end
if endsWith(id,"burst"), removed=ismember(e,306:315); end
if startsWith(id,"star_"), assert(all(starMask(removed))); starMask(removed)=false; end
if startsWith(id,"gnss_"), assert(all(gnssMask(removed))); gnssMask(removed)=false; end
stableStar=true(size(e)); stableGnss=true(size(e));
if startsWith(id,"pixel"), stableStar(event)=false; end
if id=="ar10", stableGnss(event)=false; end
currentStar=logical(o.starValid(:)); currentGnss=logical(o.gnssValid(:));
assert(isequal(currentStar(stableStar),starMask(stableStar)));
assert(isequal(currentGnss(stableGnss),gnssMask(stableGnss)));
assert(isequaln(o.starQ(:,stableStar),b.starQ(:,stableStar)) && ...
       isequaln(o.starR(:,:,stableStar),b.starR(:,:,stableStar)) && ...
       isequaln(o.inliers(stableStar),b.inliers(stableStar)));
assert(isequaln(o.gnssZ(:,stableGnss),b.gnssZ(:,stableGnss)));
expectedDelay=zeros(size(e));
if startsWith(id,"delay"), expectedDelay(event)=str2double(extractAfter(id,"delay"))/100; end
assert(all(abs(o.arrival(:)-e-expectedDelay)<1e-10));
end

function seconds=longest_same_sign(y)
% 事件窗口内 1 Hz 各历元的同号样本数，零误差与无效样本切断区间。
if ~any(isfinite(y)), seconds=[]; return; end
s=sign(y); s(~isfinite(s))=0;
edge=[1;find(diff(s)~=0)+1;numel(s)+1];
counts=diff(edge);
seconds=max([0; counts(s(edge(1:end-1))~=0)]);
end

function row=metric_row(meta,name,unit,y)
y=y(:);
assert(all(isfinite(y)),'Nonfinite sample in %s.',name);
if isempty(y)
    values={0,NaN,NaN,NaN};
else
    values={numel(y),mean(y),sqrt(mean(y.^2)),max(abs(y))};
end
row=[meta,{name,unit},values];
end

function ang=attitude_angle(a,b)
% 对 q 和 -q 取相同的最短旋转角。弦长公式保留小转角的精度。
a=a./vecnorm(a,2,2); b=b./vecnorm(b,2,2);
flip=dot(a,b,2)<0; a(flip,:)=-a(flip,:);
ang=4*atan2(vecnorm(a-b,2,2),vecnorm(a+b,2,2))*180/pi*3600;
end

function [r,sd]=noise_stats(x,lagMax)
sd=std(x(:),0);
x=x-mean(x,1);
den=sum(x.^2,'all');
r=zeros(lagMax+1,1);
for lag=0:lagMax
    r(lag+1)=sum(x(1:end-lag,:).*x(1+lag:end,:),'all')/den;
end
end

function write_pairs(T,dest)
keys={'rep','window','metric','unit'};
base=T(T.case_id=="baseline",[keys,{'mean','rms','peak_abs'}]);
base=renamevars(base,{'mean','rms','peak_abs'},{'baseline_mean','baseline_rms','baseline_peak_abs'});
C=innerjoin(T(T.case_id~="baseline",:),base,'Keys',keys);
C.delta_mean=C.mean-C.baseline_mean;
C.delta_rms=C.rms-C.baseline_rms;
C.delta_peak_abs=C.peak_abs-C.baseline_peak_abs;
writetable(C,fullfile(dest,'paired_by_seed.csv'),'Encoding','UTF-8');
[g,S]=findgroups(C(:,{'case_id','case_name','window','metric','unit'}));
S.pairs=splitapply(@(x)sum(isfinite(x)),C.delta_mean,g);
for key=["delta_mean","delta_rms","delta_peak_abs"]
    S.(key+"_avg")=splitapply(@(x)mean(x,'omitnan'),C.(key),g);
    S.(key+"_min")=splitapply(@(x)min(x,[],'omitnan'),C.(key),g);
    S.(key+"_max")=splitapply(@(x)max(x,[],'omitnan'),C.(key),g);
end
writetable(S,fullfile(dest,'paired_summary.csv'),'Encoding','UTF-8');
end

function y=read_metric(T,ids,win,name,column)
y=nan(size(ids));
for i=1:numel(ids)
    idx=T.rep==1 & T.case_id==ids(i) & T.window==win & T.metric==name;
    assert(nnz(idx)==1);
    y(i)=T.(column)(idx);
end
end

function [fig,tiles]=new_figure(w,h,nr,nc)
fig=figure('Visible','off','Color','white','Units','pixels','Position',[80,80,w,h], ...
    'DefaultAxesFontName','Microsoft YaHei','DefaultTextFontName','Microsoft YaHei', ...
    'DefaultLegendFontName','Microsoft YaHei','DefaultAxesFontSize',10, ...
    'DefaultLineLineWidth',1.3,'DefaultAxesBox','off','DefaultAxesTickDir','out');
tiles=tiledlayout(fig,nr,nc,'TileSpacing','compact','Padding','compact');
end

function finish(fig,file)
% print/saveas 的传统路径也可用于无显示的 MATLAB 批处理。
exportgraphics(fig,file,'Resolution',180,'BackgroundColor','white');
close(fig);
end

function make_plots(dest,t,P,T,figureIds)
blue=[0.06 0.36 0.57]; orange=[0.80 0.31 0.08]; green=[0.09 0.49 0.37]; grey=[0.35 0.38 0.39];
colors=[grey;blue;orange;green];
tZoom=t(t>=295&t<=340); tFollow=t(t>=300&t<420); epochs=(300:419).';

% 图一：融合同类通道的三条曲线，底部轨道独立标出两种时间安排。
if ismember(1,figureIds)
[f,l]=new_figure(980,840,4,1);
title(l,'星敏和北斗缺测期间的融合误差（第 1 组种子）','FontSize',15,'FontName','SimHei');
fields=["att","pos","vel"]; labels=["姿态误差（角秒）","位置误差（m）","速度误差（m/s）"];
for j=1:3
    ax=nexttile(l); hold(ax,'on');
    if j==1, ids=["baseline","star_sparse","star_burst"]; else, ids=["baseline","gnss_sparse","gnss_burst"]; end
    for i=1:3
        plot(ax,tZoom,P.(ids(i)).outage.(fields(j)),'Color',colors(i,:));
    end
    xlim(ax,[295,340]); grid(ax,'on'); ylabel(ax,labels(j));
    if j==1, legend(ax,{'正常','分散缺测','连续缺测'},'Location','northeast','Orientation','horizontal'); end
end
ax=nexttile(l); hold(ax,'on');
plot(ax,301:2:319,2*ones(1,10),'x','Color',blue,'MarkerSize',7);
plot(ax,306:315,ones(1,10),'x','Color',orange,'MarkerSize',7);
plot(ax,[320,316],[2,1],'o','Color',green,'MarkerFaceColor','white','MarkerSize',7);
text(ax,320.7,2,'320 s 恢复'); text(ax,316.7,1,'316 s 恢复');
ylim(ax,[0.4,2.6]); yticks(ax,[1,2]); yticklabels(ax,{'连续缺测','分散缺测'});
xlim(ax,[295,340]); xlabel(ax,'时间（s）'); ylabel(ax,'移除时刻（×）'); grid(ax,'on');
finish(f,fullfile(dest,'01_观测缺测.png'));
end

% 图二：成功率和精度同时报告；共同有效时刻用于控制样本选择。
if ismember(2,figureIds)
ids=["pixel005","baseline","pixel020","pixel040"];
[f,l]=new_figure(1100,850,2,2);
title(l,'星点测量噪声对有效定姿率与姿态精度的影响','FontSize',15,'FontName','SimHei');
ax=nexttile(l); rates=read_metric(T,ids,"event","star_valid","mean");
bar(ax,1:4,rates,0.55,'FaceColor',blue); ylim(ax,[0,112]);
for i=1:4, text(ax,i,rates(i)+2,sprintf('%.1f%%',rates(i)),'HorizontalAlignment','center'); end
ylabel(ax,'有效定姿比例（%）'); grid(ax,'on'); pixel_axis(ax);
ax=nexttile(l); hold(ax,'on');
plot(ax,1:4,read_metric(T,ids,"event","star_sample_att","rms"),'-o','Color',blue);
plot(ax,1:4,read_metric(T,ids,"event","star_common4_att","rms"),'--x','Color',orange);
plot(ax,1:4,read_metric(T,ids,"event","fused_att","rms"),'-s','Color',green);
plot(ax,1:4,read_metric(T,ids,"after","fused_att","rms"),':d','Color',grey);
pixel_axis(ax); ylabel(ax,'姿态均方根误差（角秒）'); grid(ax,'on');
legend(ax,{'单帧：本档有效','单帧：四档共同有效','融合：事件期间','融合：恢复后 60 s'},'Location','northwest');
% 跨种子的对数坐标同时容纳常规测量误差和错误星图匹配产生的大误差。
ax=nexttile(l,[1 2]); hold(ax,'on');
for id=["baseline","pixel040"]
    vals=zeros(5,2);
    for rep=1:5
        select=T.rep==rep & T.case_id==id & T.window=="event";
        vals(rep,1)=T.rms(select & T.metric=="star_sample_att");
        vals(rep,2)=T.rms(select & T.metric=="fused_att");
    end
    if id=="baseline", color=grey; labels={'基线：单帧','基线：融合'};
    else, color=orange; labels={'0.40 像素：单帧','0.40 像素：融合'}; end
    plot(ax,1:5,vals(:,1),'--o','Color',color,'DisplayName',labels{1});
    plot(ax,1:5,vals(:,2),'-s','Color',color,'DisplayName',labels{2});
end
set(ax,'YScale','log'); xlim(ax,[0.8 5.2]); xticks(ax,1:5); grid(ax,'on');
xlabel(ax,'配对组号'); ylabel(ax,'姿态均方根误差（角秒，对数刻度）');
title(ax,'五组复验中同一时段的单帧与融合误差（上排为第 1 组）');
legend(ax,'Location','northeast','NumColumns',2);
finish(f,fullfile(dest,'02_星敏图像噪声.png'));
end

% 图三：按卫星分别计算时移积，避免拼接不同卫星或跨边界相关。
if ismember(3,figureIds)
[f,l]=new_figure(1180,920,4,2);
title(l,'白噪声和时间相关噪声下的定位误差（第 1 组种子）','FontSize',15,'FontName','SimHei');
ax=nexttile(l); hold(ax,'on');
plot(ax,0:30,P.baseline.acf,'Color',blue); plot(ax,0:30,P.ar10.acf,'Color',orange);
plot(ax,0:30,exp(-(0:30)/10),'--','Color',grey);
grid(ax,'on'); ylabel(ax,'自相关系数'); xlabel(ax,'滞后（s）'); 
title(ax,'0–600 s 完整测距噪声序列'); legend(ax,{'白噪声','相关噪声','理论指数衰减'},'Location','northeast');
ax=nexttile(l); hold(ax,'on');
for i=1:2
    id=["baseline","ar10"]; inp=P.(id(i));
    plot(ax,epochs,vecnorm(inp.rawXYZ,2,2),'--','Color',colors(i+1,:));
    plot(ax,tFollow,vecnorm(inp.fusedXYZ,2,2),'Color',colors(i+1,:));
end
grid(ax,'on'); xlim(ax,[300,420]); title(ax,'三维位置误差（360 s 恢复）'); xlabel(ax,'时间（s）'); ylabel(ax,'误差长度（m）');
xline(ax,360,':','Color',grey,'HandleVisibility','off');
legend(ax,{'白：北斗','白：融合','相关：北斗','相关：融合'},'NumColumns',2,'Location','northeast');
xyz=["X","Y","Z"];
for j=1:3
    for sensor=["rawXYZ","fusedXYZ"]
        ax=nexttile(l); hold(ax,'on');
        if sensor=="rawXYZ", scale=epochs; channel='北斗单独定位'; else, scale=tFollow; channel='融合定位'; end
        plot(ax,scale,P.baseline.(sensor)(:,j),'Color',blue);
        plot(ax,scale,P.ar10.(sensor)(:,j),'Color',orange);
        yline(ax,0,':','Color',grey,'HandleVisibility','off');
        xline(ax,360,':','Color',grey,'HandleVisibility','off');
        grid(ax,'on'); xlim(ax,[300,420]); ylabel(ax,xyz(j)+" 误差（m）");
        if j==1, title(ax,[channel ' · ECI 分量']); end
        if j==3, xlabel(ax,'时间（s）'); end
    end
end
finish(f,fullfile(dest,'03_伪距时间相关性.png'));
end

% 图四：同一姿态测量分别与采集、到达时刻真值对齐，再看滤波误差。
if ismember(4,figureIds)
ids=["baseline","delay005","delay010","delay020"]; delays=[0,.05,.10,.20];
[f,l]=new_figure(1120,510,1,2);
title(l,'星敏观测延迟与姿态误差（第 1 组种子）','FontSize',15,'FontName','SimHei');
ax=nexttile(l); hold(ax,'on');
metrics=["star_sample_att","star_arrival_att","fused_att"];
for i=1:3
    plot(ax,delays,read_metric(T,ids,"event",metrics(i),"rms"),'-o','Color',colors(i+1,:));
end
plot(ax,delays,norm([.05,-.03,.08])*3600*delays,'--','Color',grey);
grid(ax,'on'); xticks(ax,delays); xlabel(ax,'观测延迟（s）'); ylabel(ax,'角度（角秒）');
title(ax,'事件期间的均方根误差与理论转角');
legend(ax,{'按采样时刻比较','按到达时刻比较','融合姿态误差','理论转角'},'Location','northwest');
ax=nexttile(l); hold(ax,'on');
for i=1:4, plot(ax,tFollow,P.(ids(i)).followupAtt,'Color',colors(i,:)); end
xline(ax,360,':','Color',grey,'HandleVisibility','off'); title(ax,'姿态误差随时间变化（360 s 恢复）');
grid(ax,'on'); xlim(ax,[300,420]); xlabel(ax,'时间（s）'); ylabel(ax,'融合姿态误差（角秒）');
legend(ax,{'0 s','0.05 s','0.10 s','0.20 s'},'Location','northwest','NumColumns',2);
finish(f,fullfile(dest,'04_星敏观测延迟.png'));
end
end

function pixel_axis(ax)
xlim(ax,[0.5,4.5]); xticks(ax,1:4); xticklabels(ax,{'0.05','0.10','0.20','0.40'});
xlabel(ax,'每轴星点测量噪声标准差（像素）');
end
