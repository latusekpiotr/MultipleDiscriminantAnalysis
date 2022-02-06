% 1.1
fv1 = [ 1 0 0;
1 1 1 ]';
fv2 = [ 1 1 0;
1 0 1 ]';
features = [ fv1(2:3, :)
fv2(2:3, :) ];
a = perceptron(fv1, fv2, 0.01,
0.1);

% 1.2
fv1 = [ fv1(1, :)' fv1(2, :)'
fv1(3, :)' (fv1(2, :) .*
fv1(3, :))' ]';
fv2 = [ fv2(1, :)' fv2(2, :)'
fv2(3, :)' (fv2(2, :) .*
fv2(3, :))' ]';
features = [ fv1(2:4, :)
fv2(2:4, :) ];
a = perceptron(fv1, fv2, 0.01,
0.05);

% 1.3
[X, Y] = meshgrid(-2:1:2, -2:1:2);
Z = - ((a(1) + a(2) .* X + a(3) .*
Y) ./ a(4));
figure;
surf(X, Y, Z); hold on;
scatter3(fv1(2, :), fv1(3, :),
fv1(4, :), 300, [1 0 0],
'filled'); hold on;
scatter3(fv2(2, :), fv2(3, :),
fv2(4, :), 300, [0 0 1],
'filled');
title('Fig 1. Hyperplane dividing
mapped data.');

% 2.2
load simple_data_PCA.mat;
figure;
plot(simple_data_PCA(:, 1),
simple_data_PCA(:, 2), 'b.'); hold
on;

% 2.3
C = cov(simple_data_PCA);
[EVectors EValues] = eig(C);

% 2.4
largestPC = EVectors(:,
find(sum(EValues) ==
max(sum(EValues))));
X = [min(simple_data_PCA(:, 1)) -
1 max(simple_data_PCA(:, 1)) + 1];
Y = (largestPC(1) .* X) ./
largestPC(2);
plot(X, Y, 'r.-');
title('Fig 2. A set of samples and
a principal component with the
highest respective eigenvalue.');

% 3.2
load simple_data_MDA.mat;
fv1 = simple_data_MDA(1:200, :);
fv2 = simple_data_MDA(201:400, :);
figure;
plot(fv1(:, 1), fv1(:, 2), 'b.');
hold on;
plot(fv2(:, 1), fv2(:, 2), 'k*');
hold on;

% 3.3
targets = ones(400, 1);
targets(201:400) = 2;
[W] = MDAMod(simple_data_MDA,
targets, 1);
X = [(min(simple_data_MDA(:, 1)) -
1) (max(simple_data_MDA(:, 1)) +
1)];
Y = (W(1) .* X) ./ W(2);
plot(X, Y, 'r.-');
title('Fig 3. A set of samples
from two classes and a component
found by MDA which discriminates
between them.');

% 3.4
figure;
plot(X, Y, 'r.-'); hold on;
projectedPoints = (W' *
simple_data_MDA')' * W';
scatter(projectedPoints(1:200, 1),
projectedPoints(1:200, 2), 10, [0
0 1], 'filled'); hold on;
scatter(projectedPoints(201:400,
1), projectedPoints(201:400, 2),
10, [0 0 0], 'filled'); hold on;
title('Fig 4. The same data and
component with data projected on
it.');

% 3.5
load iris.mat;

% 3.6
[W] = MDAMod(X, y, 2);
projectedData = (W' * X')';
figure;
plot(projectedData(find(y == 1),
1), projectedData(find(y == 1),
1), 'b.'); hold on;
plot(projectedData(find(y == 2),
1), projectedData(find(y == 2),
1), 'k*'); hold on;
plot(projectedData(find(y == 3),
1), projectedData(find(y == 3),
1), 'go'); hold on;
title('Fig 5. 2 features extracted
by MDA from 4 dimensional data.');
