% Classificação multiclase - RP

% Carregamento dos dados
clear; clc; close all;
load('../datasets/divisao.mat', 'XA', 'y_mul')
X = XA;
y = y_mul;
clear XA y_mul

% Criação da rede
seed = 42;
rng(seed) % random generator
H = 5;
optmizer = 'trainrp';
net = feedforwardnet(H, optmizer);
net.layers{2}.transferFcn = 'tansig';   % neurônios tanh na camada de saída (padrão: purelin)

% configuração e inicialização dos pesos e bias
net = configure(net,X,y); 
net.iw{1} = inicializaPesos(5,36,H,'caloba1');
net.lw{2,1} = inicializaPesos(5,5,H,'caloba1');
net.b{1} = inicializaPesos(5,1,H,'caloba1'); 
net.b{2} = inicializaPesos(5,1,H,'caloba1');

% divisão do dataset
net.divideFcn = 'divideblock';
net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 0/100;

% Parâmetros gerais do treinamento
net.trainParam.show = 1;
net.trainParam.epochs = 1000;
net.trainParam.goal = 0;
net.trainParam.max_fail = 50;
net.trainParam.showWindow = true;

% Parâmetros específicos Resilient Backpropagation
net.trainParam.delt_inc = 1.2;
net.trainParam.delt_dec = 0.5;
net.trainParam.delta0 = 0.07;
net.trainParam.deltamax = 50;

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
figure, confusionchart(C, 1:5)

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