function p = inicializaPesos(n,m,N,est)
% INICIALIZAPESOS Inicializa de modo uniforme pesos e bias 
% conforme estratédias diversas
% n,m: tamanho da matriz retornada
% N:   número de neurônios na camada
% est: 'lecum':     LeCum (1989): l = 2.4*N 
%      'caloba1':    slides Caloba: l = 0.2
%      'caloba2':    slides Caloba: l = sqrt(3/N)
%      'randsmall': Matlab randsmall
%      'rands':     Matlab symetric rand (-1 +1)

switch est
    case 'lecum'
        a = N*2.4;
        p = -a + 2*a.*rand(n,m);
    case 'caloba1'
        a = 0.2;
        p = -a + 2*a.*rand(n,m);
    case 'caloba2'
        a = sqrt(3/N);
        p = -a + 2*a.*rand(n,m);
    case 'randsmall'
        p = randsmall(n,m);
    case 'rands'
        p = rands(n,m);
end

