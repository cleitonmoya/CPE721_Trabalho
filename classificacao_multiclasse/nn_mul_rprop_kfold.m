% Classificação multiclasse - RPROP - K-fold

% Carrega o dataset
clear; close all;
load('../datasets/divisao.mat', 'XA', 'y_mul', 'y_mul2')
X = XA;
y = y_mul;
clear XA y_mul

% Separação de X em treinamento e validação (k-fold)
N = length(X);
K = 10;
seed = 42;
rng(seed) % random generator
cv = cvpartition(N,'Kfold',K);

% Hiperparâmetros
p1 = [1.1 1.2 1.3];
p2 = [0.5 0.7 0.8];

% vetores auxiliares
acc_m = zeros(length(p1),length(p2));    % acurácia média de cada modelo m
std_m = zeros(length(p1),length(p2));    % desvio padrão da acurácia de cada modelo
tr_m = cell(length(p1),length(p2));      % resultados do treinamento para cada modelo

for i = 1:length(p1)
       
    for j = 1:length(p2)

        C_total = zeros(5,5);   % matriz de confusão
        acc_k = zeros(1,K);     % acurácia de cada fold 

        % Loop do k-fold
        for k=1:K
            % Datasets de treinamento e validação
            X_tr = X(:,training(cv,k));
            y_tr = y(:, training(cv,k));
            X_vl = X(:,test(cv,k));
            y_vl = y(:, test(cv,k));

            % Junção dos data-sets para entrada do modelo
            X2 = [X_tr, X_vl];
            y2 = [y_tr, y_vl];

            % Criação da rede
            H = 5;
            optmizer = 'trainrp';
            net = feedforwardnet(H, optmizer);
            net.layers{2}.transferFcn = 'tansig';   % Seta o último neurônio como tangente hiperbólico
            
            % Configuração e inicialização dos pesos e bias            
            net = configure(net,X,y);
            net.iw{1} = inicializaPesos(5,36,H,'caloba1');
            net.lw{2,1} = inicializaPesos(5,5,H,'caloba1');
            net.b{1} = inicializaPesos(5,1,H,'caloba1'); 
            net.b{2} = inicializaPesos(5,1,H,'caloba1');
            
            % Divisão do dataset
            net.divideFcn = 'divideblock';
            net.divideParam.trainRatio = (100-K)/100;
            net.divideParam.valRatio = K/100;
            net.divideParam.testRatio = 0;
            
            % Parâmetros gerais de treinamento
            net.trainParam.show = 1;
            net.trainParam.epochs = 1000;
            net.trainParam.goal = 0;
            net.trainParam.max_fail = 50;
            net.trainParam.showWindow = true;

            % Parâmetros específicos Resilient Backpropagation
            net.trainParam.delt_inc = p1(i);
            net.trainParam.delt_dec = p2(j);
            net.trainParam.delta0 = 0.07;
            net.trainParam.deltamax = 50;

            % Treinamento
            [net,tr] = train(net,X2,y2);
            tr_m{i,j} = tr;

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
        acc_m(i,j) = mean(acc_k);
        std_m(i,j) = std(acc_k);
        fprintf('mu0 = %d, mu_f = %d: Acurácia média %d-folds: (%.4f ± %.4f)\n', p1(i), p2(j), K, acc_m(i,j), std_m(i,j))
    end
end

% Modelo com a melhor performance (validação)
max_acc = max(acc_m, [], 'all');
[i,j] = find(acc_m == max_acc);
i=i(1);
j=j(1);
tr = tr_m{i,j};
fprintf('Melhores parâmetros: p1= %d, p2 = %d, acc=(%.4f ± %.4f)\n', p1(i), p2(j), acc_m(i,j), std_m(i,j))

% Evolução do treinamento - Último fold do melhor modelo
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