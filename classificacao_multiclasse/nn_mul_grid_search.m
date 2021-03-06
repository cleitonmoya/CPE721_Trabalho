% Grid Search - Classificação Multiclasse

clear; close all; clc;

% Carrega o dataset
load('../datasets/divisao.mat', 'XD', 'y_mul', 'XD_te', 'y_mul_te')
X = XD;
y = y_mul;
X_te = XD_te;
y_te = y_mul_te;
clear XD y_mul XD_te y_mul_te
[n_feat, ~] = size(X);  % número de features
[n_out, ~] = size(y);   % número de saídas

% Separação de X em treinamento e validação (k-fold)
N = length(X);
K = 10;
rng(42)  % random generator
cv = cvpartition(N,'Kfold',K);

% Hiperparâmetros principais
O = {'traingd', 'trainlm', 'trainbfg', 'trainrp'};  % otimizador
I = {'default', 'caloba1', 'caloba2'};              % método de inicialização dos pesos e bias
H = [3, 4, 5, 6, 7];                                % número de neurônios na camada oculta

% Estruturas para a melhor e pior solução encontrada na busca
bs = struct('o', [], 'i', [], 'h', [], 'p1', [], 'p2', [], 'acc', 0, 'sd', 0, 'net', [], 'tr', [], 'te', []);
ws = struct('o', [], 'i', [], 'h', [], 'p1', [], 'p2', [], 'acc', 1, 'sd', 0, 'net', [], 'tr', [], 'te', []);

% Loop dos otimizadores
it = 1;

tiBusca = tic;
for o = 1:numel(O)
    
    opt = O{o};
    fprintf('\nAvaliando o otimizador %s\n', opt);

    % Hiperparâmetros de cada otimizador
    P1 = [];
    P2 = [];
    switch opt
        case 'traingd'
            P1 = [0.05, 0.1, 0.2];      % taxa de aprendizado
            P2 = [1];
        case 'trainlm'
            P1 = [0.001, 0.005, 0.01];  % mu0
            P2 = [10, 100, 500];        % incremento / decremento
        case 'trainbfg'
            P1 = [0.0001 0.001 0.01];
            P2 = [0.001 0.01 0.1];
        case 'trainrp'
            P1 = [1.1 1.2 1.3];
            P2 = [0.5 0.7 0.8];
    end
    
    % Loop dos métodos de inicialização
    for p = 1:numel(I)
        
        ini = I{p};
        fprintf('\tavaliando inicialização %s\n', ini);

        % Loop do número de neurônios da camada oculta
        for q = 1:length(H)
            h = H(q);

            % Loop do parâmetro p1
            for i = 1:length(P1)
                p1 = P1(i);

                % Loop do parâmetro p2
                for j = 1:length(P2)
                    p2 = P2(j);
                    C_total = zeros(2,2);   % matriz de confusão
                    acc_k = zeros(1,K);     % acurácia de cada fold 

                    if mod(it,10)==0
                        fprintf('\titeração: %d, melhor acc=%.4f, te=%.0fs\n', it, bs.acc, toc(tiBusca))
                    end

                    % Loop do k-fold
                    tiFold = tic;
                    for k=1:K
                        % Datasets de treinamento e validação
                        X_tr = X(:,training(cv,k));
                        y_tr = y(:,training(cv,k));

                        X_vl = X(:,test(cv,k));
                        y_vl = y(:,test(cv,k));

                        % Junção dos data-sets para entrada do modelo
                        X2 = [X_tr, X_vl];
                        y2 = [y_tr, y_vl];

                        % Criação da rede
                        net = feedforwardnet(h, opt);
                        net.layers{2}.transferFcn = 'tansig'; % Seta o último neurônio como tangente hiperbólico

                        % Configuração e inicialização dos pesos e bias
                        net = configure(net,X,y);
                        if ~strcmp(ini, 'default')
                            net.iw{1} = inicializaPesos(h,n_feat,h,ini);
                            net.lw{2,1} = inicializaPesos(n_out,h,h,ini);
                            net.b{1} = inicializaPesos(h,1,h,ini); 
                            net.b{2} = inicializaPesos(n_out,1,h,ini);
                        end

                        % Divisão do dataset
                        net.divideFcn = 'divideblock';
                        net.divideParam.trainRatio = (100-K)/100;
                        net.divideParam.valRatio = K/100;
                        net.divideParam.testRatio = 0;

                        % Parâmetros gerais do treinamento
                        net.trainParam.show = 1;    % número de lotes por época
                        net.trainParam.goal = 0;
                        net.trainParam.showWindow = false;
                        net.trainParam.epochs = 100;
                        net.trainParam.max_fail = 10;

                        % Parâmetros específicos de cada otimizador
                        switch opt
                            case 'traingd'             
                                net.trainParam.lr = p1;
                                net.trainParam.epochs = 1000;
                                net.trainParam.max_fail = 100;
                            case 'trainlm'
                                net.trainParam.mu = p1;
                                net.trainParam.mu_dec = p2/1000;
                                net.trainParam.mu_inc = p2;

                            case 'trainbfg'
                                net.trainParam.alpha = p1;
                                net.trainParam.beta = p2;
                            case 'trainrp'
                                net.trainParam.delt_inc = p1;
                                net.trainParam.delt_dec = p2;
                                net.trainParam.delta0 = 0.07;
                                net.trainParam.deltamax = 50;
                        end

                        % Treinamento
                        [net,tr] = train(net,X2,y2);

                        % Erro de classificação (na validação)
                        X_vl2 = X2(:,tr.valInd);
                        y_vl2 = y2(:,tr.valInd);
                        g = net(X_vl2); % predição
                        [error, ~, ~, ~] = confusion(heaviside(y_vl2),heaviside(g));
                        acc_k(k) = 1-error;
                    end
                    tFold = toc(tiFold);

                    % Acurácia - média do modelo para K-folds:
                    acc_m = mean(acc_k);
                    std_m = std(acc_k);
                    
                    % Se encontrou melhor solução, atualiza 'bs'
                    if acc_m > bs.acc
                        bs.o = opt;
                        bs.i = ini;
                        bs.h = h;
                        bs.p1 = p1;
                        bs.p2 = p2;
                        bs.acc = acc_m;
                        bs.sd = std_m;
                        bs.net = net;
                        bs.tr = tr;
                        bs.te = tFold;
                    end
 
                    % Se encontrou pior solução, atualiza 'ws'
                    if acc_m < ws.acc
                        ws.o = opt;
                        ws.i = ini;
                        ws.h = h;
                        ws.p1 = p1;
                        ws.p2 = p2;
                        ws.acc = acc_m;
                        ws.sd = std_m;
                        ws.net = net;
                        ws.tr = tr;
                        ws.te = tFold;
                    end
                    it = it+1;
                end
            end
        end
    end
end
tBusca = toc(tiBusca);

%%
% Modelos com a melhor e pior performance performance (validação)
fprintf('\nBusca concluída em %.0fs, %d modelos avaliados.\n', tBusca, it)
fprintf('Melhor modelo encontrado: opt=%s, ini=%s, h=%d, p1=%.4f, p2=%.4f, acc=(%.4f ± %.4f), te=%.0fs\n', bs.o, bs.i, bs.h, bs.p1, bs.p2, bs.acc, bs.sd, bs.te)
fprintf('Pior modelo encontrado: opt=%s, ini=%s, h=%d, p1=%.4f, p2=%.4f, acc=(%.4f ± %.4f), te=%.0fs\n', ws.o, ws.i, ws.h, ws.p1, ws.p2, ws.acc, ws.sd, ws.te)

%%
% Evolução do treinamento no último fold do melhor e pior modelo
tr_bs = bs.tr;
tr_ws = ws.tr;

figure()
title('Performance - classif. 5 classes')

otimizadores = {'GD', 'LM', 'BFGS', 'RPROP'};
[idx_mod_ws, ~] = ismember(O, ws.o);
otm_ws = otimizadores(idx_mod_ws);
otm_ws = otm_ws{1};

[idx_ini_ws, ~] = ismember(I, ws.i);
ini_ws = find(idx_ini_ws);

[idx_mod_bs, ~] = ismember(O, bs.o);
otm_bs = otimizadores(idx_mod_bs);
otm_bs = otm_bs{1};

[idx_ini_bs, ~] = ismember(I, bs.i);
ini_bs = find(idx_ini_bs);

% Melhor modelo
subplot(2,1,1)
semilogy(tr_bs.perf, 'LineWidth', 1)
hold on
semilogy(tr_bs.vperf, 'LineWidth', 1)
[vperf_min, it_min] = min(tr_bs.vperf);
subtitle(compose('Melhor modelo: H=%d, %s, inic. tipo %d', bs.h, otm_bs, ini_bs))
xline(it_min,':')
yline(vperf_min, ':')
ylabel('mse')
legend({'Treinamento', 'Validação', 'Menor erro (valid.)'});

% Pior modelo
subplot(2,1,2)
semilogy(tr_ws.perf, 'LineWidth', 1)
hold on
semilogy(tr_ws.vperf, 'LineWidth', 1)
[vperf_min, it_min] = min(tr_ws.vperf);
subtitle(compose('Pior modelo: H=%d, %s, inic. tipo %d', ws.h, otm_ws, ini_ws))
xline(it_min,':')
yline(vperf_min, ':')
xlabel('época')
ylabel('mse')


%%
% Avaliação final do melhor modelo no conjunto de teste
net = bs.net;

% Matriz de confusão e erro
g = net(X_te); % predição
[error, C, ~, ~] = confusion(heaviside(y_te),heaviside(g));
acc = 1-error;
fprintf('Acurácia final no conjunto de teste: %f\n',acc)

figure()
cm = confusionchart(C,1:5);
cm.RowSummary = 'row-normalized';
title(compose('Classif. binária - Acurácia final: %.1f%% (conjunto de teste)',acc*100));
xlabel('Classe predita')
ylabel('Classe verdadeira')

%%
% Arquitetura da rede
view(net)

% Diagrama de Hinton
figure()
plotwb(net)