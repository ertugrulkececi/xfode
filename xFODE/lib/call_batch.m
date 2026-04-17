%%
function [mini_batch_inputs, targets, u_mini_batch] = call_batch(batch_inputs, U, batch_targets,iter,mbs)

mini_batch_inputs = batch_inputs(:, :, ((iter-1)*mbs)+1:(iter*mbs));
targets = batch_targets(:, :, ((iter-1)*mbs)+1:(iter*mbs));
u_mini_batch = U(:, :, ((iter-1)*mbs)+1:(iter*mbs));

end
