% Classificação binária - GD - K-fold

% Carrega o dataset
clear; close all; clc;
load('../datasets/divisao.mat', 'XA', 'y_bin')
X = XA;
y = y_bin;
clear XA y_bin

% Separação de X em treinamento e validação (k-fold)
N = length(X);
K = 10;
seed = 42;
rng(seed) % random generator
cv = cvpartition(N,'Kfold',K);

H = [2, 3, 4, 5, 6, 7, 8];
L = [0.05, 0.1, 0.2];
P = H;

acc_m = zeros(1,length(P)); % acurácia média de cada modelo m
std_m = zeros(1,length(P)); % desvio padrão da acurácia de cada modelo
tr_m = cell(length(P));     % resultados do treinamento para cada modelo

for j = 1:length(P)
    
    p = P(j);
    
    C_total = zeros(2,2);   % matriz de confusão
    acc_k = zeros(1,K);     % acurácia de cada fold 
    
    % Loop do k-fold
    for k=1:K
        % Datasets de treinamento e validação
        X_tr = X(:,training(cv,k));
        y_tr = y(training(cv,k));

        X_vl = X(:,test(cv,k));
        y_vl = y(test(cv,k));

        % Junção dos data-sets para entrada do modelo
        X2 = [X_tr, X_vl];
        y2 = [y_tr, y_vl];

        % Criação da rede
        hiddenLayerSize = p;
        optmizer = 'traingd';
        net = feedforwardnet(hiddenLayerSize, optmizer);

        % Parâmetros da rede
        net.layers{2}.transferFcn = 'tansig'; % Seta o último neurônio como tangente hiperbólico
        net.trainParam.show = 1;
        net.trainParam.lr = 0.1;
        net.trainParam.epochs = 500;
        net.trainParam.goal = 1e-1;
        net.trainParam.max_fail = 5;
        net.trainParam.showWindow = 0;
        net.divideFcn = 'divideblock';

        net.divideParam.trainRatio = (100-K)/100;
        net.divideParam.valRatio = K/100;
        net.divideParam.testRatio = false;

        % Treinamento
        [net,tr] = train(net,X2,y2);
        tr_m{j} = tr;
        
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
    acc_m(j) = mean(acc_k);
    std_m(j) = std(acc_k);
    fprintf('P = %d: Acurácia média %d-folds: (%.4f ± %.4f)\n', p, K, acc_m(j), std_m(j))
end

% Modelo com melhor performance (validação)
[~, m_max_acc] = max(acc_m);
fprintf('\nMelhor parâmetro: p=%d, acc=(%.4f ± %.4f)\n', P(m_max_acc), acc_m(m_max_acc), std_m(m_max_acc))

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

