data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

X1 = X(:,1);
X2 = X(:,2);

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

Xreg = out;

initial_theta = zeros(size(Xreg, 2), 1);
size(initial_theta)

thetaReg = (initial_theta(2:size(initial_theta),:));

size(thetaReg)