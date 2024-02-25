import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset
from dgl.nn import GATConv
import matplotlib.pyplot as plt

# 假设每个学生节点的特征是一个一维张量，表示成绩，总共有20个成绩
num_students = 100  # 假设有100个学生
num_edges = 200  # 用于构建随机图的边的数量
num_grades = 20  # 每个学生有20次考试成绩
num_heads = [8, 1]  # GAT模型中每一层的注意力头数

# 生成随机图数据
g = dgl.rand_graph(num_students, num_edges)
g = dgl.add_self_loop(g)

# 随机初始化学生节点的特征，即成绩
student_grades = torch.randint(0, 101, (num_students, num_grades), dtype=torch.float)  # 成绩范围在0到100之间
print(f"student_grades[1]: {student_grades[1]}")
g.ndata['feat'] = student_grades.unsqueeze(1)  # 增加一维，表示成绩的特征是一个张量


class GAT(nn.Module):

    def __init__(
            self, in_dim, hidden_dim, out_dim, num_heads, dropout,
            residual=False, activation=None):
        """GAT模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: List[int] 每一层的注意力头数，长度等于层数
        :param dropout: float Dropout概率
        :param residual: bool, optional 是否使用残差连接，默认为False
        :param activation: callable, optional 输出层激活函数
        :raise ValueError: 如果层数（即num_heads的长度）小于2
        """
        super().__init__()
        num_layers = len(num_heads)
        if num_layers < 2:
            raise ValueError('层数至少为2，实际为{}'.format(num_layers))
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(
            in_dim, hidden_dim, num_heads[0], dropout, dropout, residual=residual, activation=F.elu
        ))
        for i in range(1, num_layers - 1):
            self.layers.append(GATConv(
                num_heads[i - 1] * hidden_dim, hidden_dim, num_heads[i], dropout, dropout,
                residual=residual, activation=F.elu
            ))
        self.layers.append(GATConv(
            num_heads[-2] * hidden_dim, out_dim, num_heads[-1], dropout, dropout,
            residual=residual, activation=activation
        ))

    def forward(self, g, h):
        """
        :param g: DGLGraph 同构图
        :param h: tensor(N, d_in) 输入特征，N为g的顶点数
        :return: tensor(N, d_out) 输出顶点特征，K为注意力头数
        """
        for i in range(len(self.layers) - 1):
            h = self.layers[i](g, h).flatten(start_dim=1)  # (N, K, d_hid) -> (N, K*d_hid)
        h = self.layers[-1](g, h).mean(dim=1)  # (N, K, d_out) -> (N, d_out)
        return h


# 初始化 GAT 模型
model = GAT(in_dim=num_grades, hidden_dim=64, out_dim=1, num_heads=num_heads, dropout=0.6)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

def custom_loss(logits, g, specific_index):
    # 获取与特定学生相连的节点索引
    neighbor_indices = g.successors(specific_index)
    # 计算邻居节点的平均成绩
    neighbor_avg = torch.mean(logits[neighbor_indices])

    # 获取特定学生的当前成绩
    specific_avg = torch.mean(logits[specific_index])

    # 定义损失函数，使得特定学生的成绩上升5分时，相邻节点的平均成绩也上升
    loss = (neighbor_avg - specific_avg) ** 2

    return loss

num_epochs = 100
losses = []
for epoch in range(num_epochs):
    # 前向传播
    logits = model(g, g.ndata['feat'])
    
    # 计算损失
    loss = custom_loss(logits, g, specific_index=0)  # 假设特定学生的索引为0
    losses.append(loss.item())
    
    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 输出当前训练进度
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 提取除了特定学生之外的其他学生的预测输出
specific_index = 0
other_students_logits = torch.cat((logits[:specific_index], logits[specific_index+1:]), dim=0)
avg_other_students = torch.mean(other_students_logits)
print("其他 99 个学生的平均成绩:", avg_other_students.item())
print("2號 学生的成绩:", logits[2])
print("logits shape", logits.shape) # shape torch.Size([100, 1])
print("logits[:specific_index]", logits[:specific_index]) #  tensor([], size=(0, 1), grad_fn=<SliceBackward0>)
print("logits[:specific_index]", logits[:specific_index].shape) #  torch.Size([0, 1]
# # 训练模型
# num_epochs = 100
# losses = []    
# for epoch in range(num_epochs):
#     # 前向传播
#     logits = model(g, g.ndata['feat'])
    
#     # 计算损失
#     loss = torch.mean(logits)
#     losses.append(loss.item())
    
#     # 反向传播与优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     # 输出当前训练进度
#     if epoch % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 可视化损失曲线
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()
