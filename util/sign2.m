function y = sign2(x)
% SIGN2 signal(x) = {+1, se x>=0
%                    -1, se x<0 

    x1 = (x>=0)*1;
    x2 = (x<0)*-1;
    y=x1+x2;
end