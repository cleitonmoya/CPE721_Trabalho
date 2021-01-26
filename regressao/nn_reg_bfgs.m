% Regressão - BFGS

% Carregamento dos dados
clear; clc; close all;
load('../datasets/divisao.mat', 'XB', 'y_reg')
X = XB;
y = y_reg;
clear XB y_reg
[n_feat, ~] = size(X);  % número de features
[n_out, ~] = size(y);   % número de saídas

% Criação da rede
seed = 42;
rng(seed) % random generator
h = 5;
optmizer = 'trainbfg';
net = feedforwardnet(h, optmizer);
%net.layers{2}.transferFcn = 'purelin'; % neurônio linear na camada de saída

ini='caloba2'
% Configuração e inicialização dos pesos e bias
net = configure(net,X,y); 
% net.iw{1} = inicializaPesos(h,n_feat,h,ini);
% net.lw{2,1} = inicializaPesos(n_out,h,h,ini);
% net.b{1} = inicializaPesos(h,1,h,ini); 
% net.b{2} = inicializaPesos(n_out,1,h,ini);

% Divisão do dataset
net.divideFcn = 'divideblock';
net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 0/100;

% Parâmetros gerais do treinamento
net.trainParam.show = 1;
net.trainParam.epochs = 100;
net.trainParam.goal = 0;
net.trainParam.max_fail = 10;
net.trainParam.showWindow = true;

% Parâmetros específicos do BFGS
net.trainParam.alpha = 0.0001;
net.trainParam.beta = 0.001;
% net.trainParam.delta = 0.01; % initial step size
% net.trainParam.low_lim = 0.1;
% net.trainParam.up_lim = 0.5;

% Treinamento
[net,tr] = train(net,X,y);

% Acurácia
fprintf('RMSE: %.4f\n',sqrt(tr.best_vperf))

% Evolução do treinamento
[vperf_min, it_min] = min(tr.vperf);
semilogy(tr.perf, 'LineWidth', 1)
hold on
semilogy(tr.vperf, 'LineWidth', 1)
xline(it_min,':')
yline(vperf_min, ':')
xlabel('Iteração')
ylabel('Erro quadrático médio')
legend({'Treinamento', 'Validação', 'Melhor'});
