% Regressão - Gradiente Descendente

% Carregamento dos dados
clear; clc; close all;
load('../datasets/divisao.mat', 'XA', 'y_reg')
X = XA;
y = y_reg;
clear XA y_reg

% Criação da rede
seed = 42;
rng(seed) % random generator
H = 5;
optmizer = 'traingd';
net = feedforwardnet(H, optmizer);
net.layers{2}.transferFcn = 'purelin'; % neurônio linear na camada de saída

% Configuração e inicialização dos pesos e bias
net = configure(net,X,y); 
net.iw{1} = inicializaPesos(5,36,H,'caloba1');
net.lw{2,1} = inicializaPesos(1,5,H,'caloba1');
net.b{1} = inicializaPesos(5,1,H,'caloba1'); 
net.b{2} = inicializaPesos(1,1,H,'caloba1');

% Divisão do dataset
net.divideFcn = 'divideblock';
net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 0/100;

% Parâmetros gerais do treinamento
net.trainParam.show = 1;
net.trainParam.epochs = 1000;
net.trainParam.goal = 0;
net.trainParam.max_fail = 100;
net.trainParam.showWindow = true;

% Parâmetros específicos Gradiente Descendente
net.trainParam.lr = 0.01;

% Treinamento
[net,tr] = train(net,X,y);

% Acurácia
fprintf('RMSE: %.4f\n',sqrt(tr.best_vperf))

% Evolução do treinamento
[vperf_min, it_min] = min(tr.vperf);
plot(tr.perf, 'LineWidth', 1)
hold on
plot(tr.vperf, 'LineWidth', 1)
xline(it_min,':')
yline(vperf_min, ':')
xlabel('Iteração')
ylabel('Erro quadrático médio')
legend({'Treinamento', 'Validação', 'Melhor'});
