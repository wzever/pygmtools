import torch
import pygmtools as pygm
pygm.BACKEND = 'pytorch'
_ = torch.manual_seed(1)

# Generate a batch of isomorphic graphs
batch_size = 10
nodes_num = 4
feature_num = 36

X_gt = torch.zeros(batch_size, nodes_num, nodes_num)
X_gt[:, torch.arange(0, nodes_num, dtype=torch.int64), torch.randperm(nodes_num)] = 1
A1 = 1. * (torch.rand(batch_size, nodes_num, nodes_num) > 0.5)
torch.diagonal(A1, dim1=1, dim2=2)[:] = 0 # discard self-loop edges
A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
feat1 = torch.rand(batch_size, nodes_num, feature_num) - 0.5
feat2 = torch.bmm(X_gt.transpose(1, 2), feat1)
n1 = n2 = torch.tensor([nodes_num] * batch_size)

# Match by PCA-GM (load pretrained model)
X, net = pygm.a_star(feat1, feat2, A1, A2, n1, n2, return_network=True)
# Downloading to ~/.cache/pygmtools/best_genn_AIDS700nef_gcn_astar.pt...
print((X * X_gt).sum() / X_gt.sum())# accuracy
# tensor(1.)

# Pass the net object to avoid rebuilding the model agian
X = pygm.a_star(feat1, feat2, A1, A2, n1, n2, network=net)

# This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.
part_f1 = feat1[0]
part_f2 = feat2[0]
part_A1 = A1[0]
part_A2 = A2[0]
part_X_gt = X_gt[0]
part_X = pygm.a_star(part_f1, part_f2, part_A1, part_A2, return_network=False)

print(part_X.shape)
# torch.Size([4, 4])

print((part_X * part_X_gt).sum() / part_X_gt.sum())# accuracy
# tensor(1.)

# You can also use traditional heuristic methods to solve without using neural networks
X = pygm.a_star(feat1, feat2, A1, A2, n1, n2, use_net=False)
print((X * X_gt).sum() / X_gt.sum())# accuracy
# tensor(1.)            

# You may also load other pretrained weights
# However, it should be noted that each pretrained set supports different node feature dimensions
# AIDS700nef(Default): feature_num = 36
# LINUX: feature_num = 8
# Generate a batch of isomorphic graphs
batch_size = 10
nodes_num = 4
feature_num = 8

X_gt = torch.zeros(batch_size, nodes_num, nodes_num)
X_gt[:, torch.arange(0, nodes_num, dtype=torch.int64), torch.randperm(nodes_num)] = 1
A1 = 1. * (torch.rand(batch_size, nodes_num, nodes_num) > 0.5)
torch.diagonal(A1, dim1=1, dim2=2)[:] = 0 # discard self-loop edges
A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
feat1 = torch.rand(batch_size, nodes_num, feature_num) - 0.5
feat2 = torch.bmm(X_gt.transpose(1, 2), feat1)
n1 = n2 = torch.tensor([nodes_num] * batch_size)

X, net = pygm.a_star(feat1, feat2, A1, A2, n1, n2, pretrain='LINUX',return_network=True)
# Downloading to ~/.cache/pygmtools/best_genn_LINUX_gcn_astar.pt...

print((X * X_gt).sum() / X_gt.sum())# accuracy
# tensor(1.)

# When the input node feature dimension is different from the one supported by pre training, 
# you can still use the solver, but a warning is provided.
X, net = pygm.a_star(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='AIDS700nef')
# UserWarning: Pretrain AIDS700nef does not support the feature_num = 8 you entered

# You may configure your own model and integrate the model into a deep learning pipeline. For example:
net = pygm.utils.get_network(pygm.a_star, feature_num=8, filters_1=512, filters_2=256, filters_3=64, pretrain=False)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# feat1/feat2 may be outputs by other neural networks
X = pygm.a_star(feat1, feat2, A1, A2, n1, n2, network=net)
loss = pygm.utils.permutation_loss(X, X_gt)
loss.requires_grad_(True)
loss.backward()
optimizer.step()