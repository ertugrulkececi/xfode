%%
function [X0, U, targets]  = create_mini_batch(X, ux,  ahead, numexamples)

X = permute(X, [2, 3, 1]);

shuffle_idx = randperm(size(X, 2)-ahead);

X0 = dlarray(X(:, shuffle_idx));
targets = dlarray(zeros([size(X, 1) ahead, numexamples]));
U = (zeros([size(ux, 1), ahead+1, numexamples]));

for i =1:numexamples
    targets(:, :, i) = X(:, shuffle_idx(i) + 1: shuffle_idx(i) + ahead);
     U(:, :, i) = ux(:, shuffle_idx(i): shuffle_idx(i) + ahead);
end

X0 = permute(X0, [3 1 2]);

end
