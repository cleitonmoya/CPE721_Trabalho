% Carregamento dos dados
clear; clc; close all;
X = transpose(readmatrix('datasets/X_A2.txt'));
y = transpose(readmatrix('datasets/Y_bin.txt'));
N = length(X);

% Divisão do dataset para o k-fold
seed = 42;
rng(seed) % random generator

% Criação da rede
hiddenLayerSize = 5;
optmizer = 'traingd';
net = feedforwardnet(hiddenLayerSize, optmizer);

% Parâmetros da rede
net.trainParam.show = 1;
net.trainParam.lr = 0.05;
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-1;
net.trainParam.max_fail = 5;
net.trainParam.showWindow = 0;
net.divideFcn = 'divideblock';

net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 0;

% Treinamento
[net,tr] = train(net,X,y);

% Matriz de confusão e erro
X_vl = X(:,tr.valInd);
y_vl = y(:,tr.valInd);
g = net(X_vl); % predição

% Matriz de confusão e acurácia
[error, C, ~, ~] = confusion(heaviside(y_vl),heaviside(g));
acc = 1-error;
fprintf('Acurácia: %f\n',acc)
figure, confusionchart(C, [-1,1])

% Evolução do treinamento
[vperf_min, it_min] = min(tr.vperf);
figure()
plot(tr.perf, 'LineWidth', 1)
hold on
plot(tr.vperf, 'LineWidth', 1)
xline(it_min,':', 'Color', '#77AC30')
yline(vperf_min, ':', 'Color', '#77AC30')

xlabel('Iteração')
ylabel('Erro quadrático médio')
legend({'Treinamento', 'Validação', 'Melhor'});