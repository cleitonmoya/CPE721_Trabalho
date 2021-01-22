% Carregamento dos dados
clear; clc; close all;
X = transpose(readmatrix('datasets/X.txt'));
y = transpose(readmatrix('datasets/Y_cla.txt'));
N = length(X);

rng(42) % random generator
    
% Criação da rede
hiddenLayerSize = 5;
optmizer = 'traingd';
net = feedforwardnet(hiddenLayerSize, optmizer);

% Parâmetros da rede
net.trainParam.show = 1;
net.trainParam.lr = 0.1;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-1;
net.trainParam.max_fail = 5;
net.trainParam.showWindow = 0;
net.divideFcn = 'divideblock';

net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 0;

% Treinamento
[net,tr] = train(net,X,y);

X_vl = X(:,tr.valInd);
y_vl = y(:,tr.valInd);

g = net(X_vl);

eqm1 = sum(gsubtract(y_vl,g).^2)/length(g);
eqm2 = tr.best_vperf;
fprintf('Eqm 1: %.3f\n', eqm1)
fprintf('Eqm 2: %.3f\n', eqm2) 

%%
% Gráfico - EQM x Erro de classificação:

plot(tr_loss{i}, 'DisplayName', 'Treinamento')
hold on
plot(vl_loss{i}, 'DisplayName', 'Validação')
xline(x_min,'--', 'DisplayName', 'menor erro')

%legend('Treinamento','Validação')
xlabel('Iteração')
ylabel('Erro quadrático médio')
legend('show');



%%
[error, C, ~, ~] = confusion(heaviside(y_vl),heaviside(g));
figure, confusionchart(C, [-1,1])
fprintf('Acurácia: %.3f', 1-error);
