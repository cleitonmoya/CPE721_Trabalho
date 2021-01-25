% Classificação binária - Levemberg-Marquadt

% Carregamento dos dados
clear; clc; close all;
load('../datasets/divisao.mat', 'XA', 'y_bin')
X = XA;
y = y_bin;
clear XA y_bin

% Criação da rede
seed = 42;
rng(seed) % random generator
H = 7;
optmizer = 'trainlm';
net = feedforwardnet(H, optmizer);
net.layers{2}.transferFcn = 'tansig';

% configuração e inicialização dos pesos e bias
net = configure(net,X,y);
net.iw{1} = inicializaPesos(H,36,H,'caloba2');
net.lw{2,1} = inicializaPesos(1,H,H,'caloba2');
net.b{1} = inicializaPesos(H,1,H,'caloba2'); 
net.b{2} = inicializaPesos(1,1,H,'caloba2');

% divisão do dataset
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

% Parâmetros específicos Levemberg-Marquadt
net.trainParam.mu = 0.001;
net.trainParam.mu_dec = 0.1;
net.trainParam.mu_inc = 10;

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
xline(it_min,':')
yline(vperf_min, ':')
xlabel('Iteração')
ylabel('Erro quadrático médio')
legend({'Treinamento', 'Validação', 'Melhor'});
