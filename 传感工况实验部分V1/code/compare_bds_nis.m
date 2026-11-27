function compare_bds_nis()
% 回放已有观测，记录固定 ESKF 更新前的八维北斗归一化创新平方。
root=fileparts(fileparts(mfilename('fullpath')));
project=fileparts(root);
original=fullfile(project,'satellite_nav_sim');
addpath(fullfile(original,'fusion'),fullfile(original,'fusion','filters'),fullfile(original,'utils'));
data=fullfile(root,'data');
truth=load(fullfile(data,'truth.mat'),'t','r_true','v_true');
idx=make_state_index();
% 8 维北斗更新的卡方分布 99% 分位数。
threshold=20.0902350296632;
summary=table(); history=table();
for rep=1:5
    b=load(fullfile(data,sprintf('rep%02d_base.mat',rep)),'base','X0','P0','cfg');
    for name=["baseline","ar10"]
        c=load(fullfile(data,sprintf('rep%02d_%s.mat',rep,name)),'obs','out');
        o=c.obs;
        X=b.X0; P=b.P0;
        assert(isequal(o.epochs(:),o.arrival(:)) && isequal(o.epochs(:),(0:600).'));
        nis=nan(601,1);
        for k=1:numel(truth.t)-1
            t=truth.t(k);
            [X,P]=eskf_update_gyro(X,P,b.base.gyro(:,k),b.base.gyroR,idx);
            [X,P]=eskf_predict_full(X,P,t,truth.t(k+1)-t,b.cfg,idx);
            if mod(k,100)==0
                j=k/100+1;
                if o.gnssValid(j)
                    [X,P,info]=eskf_update_gnss(X,P,o.gnssZ(:,j),b.base.gnssR,idx);
                    nis(j)=info.nis;
                end
                if o.starValid(j)
                    [X,P]=eskf_update_star(X,P,o.starQ(:,j),o.starR(:,:,j),idx);
                end
                assert(norm((X.r.'-truth.r_true(k+1,:))-c.out.dpos(k+1,:))<1e-8);
                assert(abs(norm(X.v.'-truth.v_true(k+1,:))-c.out.vel(k+1))<1e-10);
            end
        end
        if rep==1 && name=="baseline"
            assert(isequal(o.epochs(2:end),(1:600).'));
            % 首组基线已记录的逐次更新最大 NIS 和稳态位置 RMS。
            assert(abs(max(nis(2:end))-6.32846702757898)<1e-6);
            k=truth.t>=60;
            assert(abs(sqrt(mean(c.out.pos(k).^2))-0.490168)<5e-7);
        end
        use=isfinite(nis); e=o.epochs(use); v=nis(use);
        history=[history;table(repmat(rep,numel(e),1),repmat(name,numel(e),1),e,v,v>threshold, ...
            'VariableNames',{'rep','condition','time_s','nis','above_threshold'})]; %#ok<AGROW>
        for window=["event","all"]
            if window=="event", pick=e>=300 & e<360; else, pick=true(size(e)); end
            row=table(rep,name,window,nnz(pick),mean(v(pick)),max(v(pick)),nnz(v(pick)>threshold),threshold, ...
                'VariableNames',{'rep','condition','window','updates','mean_nis','peak_nis','above_threshold','threshold'});
            summary=[summary; row]; %#ok<AGROW>
        end
        fprintf('Completed %d %s\n',rep,name);
    end
end
writetable(summary,fullfile(root,'results','北斗NIS对照汇总.csv'),'Encoding','UTF-8');
writetable(history,fullfile(root,'results','北斗NIS逐次更新.csv'),'Encoding','UTF-8');
end
