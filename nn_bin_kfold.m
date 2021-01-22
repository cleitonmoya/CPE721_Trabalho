% Carregamento dos dados
clear; clc; close all;
X = transpose(readmatrix('datasets/X_A2.txt'));
y = transpose(readmatrix('datasets/Y_bin.txt'));
N = length(X);

% Divisão do dataset para o k-fold
seed = 42;
rng(seed) % random generator
K = 10;
cv = cvpartition(N,'Kfold',K);

H = [2, 4, 5, 6, 8, 10, 15];
L = [0.05, 0.1, 0.2];
P = L;

acc_m = zeros(1,length(P)); % acurácia média de cada modelo m
std_m = zeros(1,length(P)); % desvio padrão da acurácia de cada modelo
tr_m = cell(1,length(P));   % resultados do treinamento para cada modelo

for j = 1:length(P)
    
    p = P(j);
    
    C_total = zeros(2,2);   % matriz de confusão
    acc_k = zeros(1,K);     % acurácia de cada fold 
    
    for k=1:K
        % Datasets de treinamento e validação
        X_tr = X(:, training(cv, k));
        y_tr = y(training(cv,k));

        X_vl = X(:, test(cv, k));
        y_vl = y(test(cv,k));

        X2 = [X_tr, X_vl];
        y2 = [y_tr, y_vl];

        % Criação da rede
        hiddenLayerSize = 4;
        optmizer = 'traingd';
        net = feedforwardnet(hiddenLayerSize, optmizer);

        % Parâmetros da rede
        net.trainParam.show = 1;
        net.trainParam.lr = p;
        net.trainParam.epochs = 300;
        net.trainParam.goal = 1e-1;
        net.trainParam.max_fail = 5;
        net.trainParam.showWindow = 0;
        net.divideFcn = 'divideblock';

        net.divideParam.trainRatio = 90/100;
        net.divideParam.valRatio = 10/100;
        net.divideParam.testRatio = 0;

        % Treinamento
        [net,tr] = train(net,X2,y2);

        % Matriz de confusão e erro
        X_vl2 = X2(:,tr.valInd);
        y_vl2 = y2(:,tr.valInd);
        g = net(X_vl2); % predição

        % Matriz de confusão
        [error, C, ~, ~] = confusion(heaviside(y_vl2),heaviside(g));
        C_total = C_total + C;
        acc_k(k) = 1-error;
        %fprintf('H = %d: Fold %d: Acurácia: %f\n', h, k, acc_k(k))
    end

    % Matriz de confusão - média do K-fold
    C = round(C_total./K);
    %figure, confusionchart(C, [-1,1])

    % Acurácia - média do modelo h para K-folds:
    tr_m{j} = tr;
    acc_m(j) = mean(acc_k);
    std_m(j) = std(acc_k);
    fprintf('P = %d: Acurácia média %d-folds: (%f ± %f)\n', p, K, acc_m(j), std_m(j))
end

% Modelo com melhor performance
[~, m_max_acc] = max(acc_m);
fprintf('\nMelhor parâmetro: p=%d, acc=(%f ± %f)\n', P(m_max_acc), acc_m(m_max_acc), std_m(m_max_acc))


% Evolução do treinamento - Último fold do melhor modelo
[vperf_min, it_min] = min(tr_m{m_max_acc}.vperf);
plot(tr_m{m_max_acc}.perf, 'LineWidth', 1)
hold on
plot(tr_m{m_max_acc}.vperf, 'LineWidth', 1)
xline(it_min,':', 'Color', '#77AC30')
yline(vperf_min, ':', 'Color', '#77AC30')

xlabel('Iteração')
ylabel('Erro quadrático médio')
legend({'Treinamento', 'Validação', 'Melhor'});
