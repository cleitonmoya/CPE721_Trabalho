% Reserva uma parte do dataset para teste

clear;

% Carregamento dos dados
X_A = transpose(readmatrix('datasets/X_A2.txt'));
X_B = transpose(readmatrix('datasets/X_B2.txt'));
X_C = transpose(readmatrix('datasets/X_C2.txt'));
X_D = transpose(readmatrix('datasets/X_D2.txt'));

Y_bin = transpose(readmatrix('datasets/Y_bin.txt'));
Y_mul = transpose(readmatrix('datasets/Y_mul.txt'));    % Codificação maximamente esparsa
Y_mul2 = transpose(readmatrix('datasets/Y_mul2.txt'));  % Categorias ordinais 
Y_reg = transpose(readmatrix('datasets/Y_reg.txt'));

% Separação dos teste
rng(42) % random generator
cv = cvpartition(Y_mul2,'Holdout',0.2);

XA = X_A(:, training(cv));
XB = X_B(:, training(cv));
XC = X_C(:, training(cv));
XD = X_D(:, training(cv));

XA_te = X_A(:, test(cv));
XB_te = X_B(:, test(cv));
XC_te = X_C(:, test(cv));
XD_te = X_D(:, test(cv));

y_bin = Y_bin(training(cv));
y_mul = Y_mul(:,training(cv));
y_mul2 = Y_mul2(:,training(cv));
y_reg = Y_reg(training(cv));

y_bin_te = Y_bin(test(cv));
y_mul_te = Y_mul(:,test(cv));
y_mul_te2 = Y_mul2(:,test(cv));
y_reg_te = Y_reg(test(cv));

% Salva os data-sets dividos
save('datasets/divisao.mat');
clear;