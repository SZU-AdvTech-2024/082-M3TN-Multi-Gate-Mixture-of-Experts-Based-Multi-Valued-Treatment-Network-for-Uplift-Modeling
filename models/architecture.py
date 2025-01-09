import numpy as np

import torch
import torch.nn as nn

# experting model
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.layer1=nn.Linear(input_dim, hidden_dim)
        self.layer2=nn.Linear(hidden_dim, output_dim)
 
    def forward(self, x):
        x=torch.relu(self.layer1(x)) # 隐藏层激活
        output = torch.softmax(self.layer2(x), dim=1)
        return output # 输出概率分布
 # Define the gating model
class Gating(nn.Module):
    def __init__(self, input_dim, num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()
 
         # Layers
        self.layer1=nn.Linear(input_dim, 128)
        self.dropout1=nn.Dropout(dropout_rate)
 
        self.layer2=nn.Linear(128, 256)
        self.leaky_relu1=nn.LeakyReLU()
        self.dropout2=nn.Dropout(dropout_rate)
 
        self.layer3=nn.Linear(256, 128)
        self.leaky_relu2=nn.LeakyReLU()
        self.dropout3=nn.Dropout(dropout_rate)
 
        self.layer4=nn.Linear(128, num_experts)
 
    def forward(self, x):
        x=torch.relu(self.layer1(x))
        x=self.dropout1(x) # 防止过拟合
 
        x=self.layer2(x)
        x=self.leaky_relu1(x)
        x=self.dropout2(x)
 
        x=self.layer3(x)
        x=self.leaky_relu2(x)
        x=self.dropout3(x)
 
        output = torch.softmax(self.layer4(x), dim=1)
        return output
# 多任务单门专家网络
class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, dropout_rate=0.1):
        super(MoE, self).__init__()
        #  self.experts=nn.ModuleList(trained_experts)
        #  num_experts=len(trained_experts)
        #  # Assuming all experts have the same input dimension
        #  input_dim=trained_experts[0].layer1.in_features
        #  self.gating=Gating(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Softmax(dim=1)  # 专家输出概率分布
            )
            for _ in range(num_experts)
        ])
        
        # 定义门控网络
        self.gating = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)  # 门控网络输出每个专家的权重
        )
 
    def forward(self, x):
         # Get the weights from the gating network
        weights=self.gating(x)
 
         # Calculate the expert outputs
        expert_outputs=torch.stack(
             [expert(x) for expert in self.experts], dim=2)
 
         # Adjust the weights tensor shape to match the expert outputs
         #weights=weights.unsqueeze(1).expand_as(outputs)
        weights=weights.unsqueeze(1)
         # Multiply the expert outputs with the weights and
         # sum along the third dimension
        output = torch.sum(expert_outputs*weights, dim=2)
        return output

class M3TN(nn.Module):
    """`
    M3TN: MULTI-GATE MIXTURE-OF-EXPERTS BASED MULTI-VALUED TREATMENT NETWORK FOR UPLIFT MODELING
    """
    def __init__(self, input_dim, h_dim, output_dim, num_experts, is_self, act_type):
        super(M3TN, self).__init__()
        self.is_self = is_self
        #self.num_tasks = num_tasks
        # MoE layer (Mixture of Experts)
        self.moe = MoE(input_dim, h_dim, output_dim, num_experts, dropout_rate=0.1)
        # representation part 引入MMoE
            # output_dim = 3
            # hidden_dim = 32
        
        # propensity for treat predict
        # self.t_fc1 = nn.Linear(h_dim, h_dim)
        # self.t_logit = nn.Linear(h_dim, 1)

        
        # control net
        self.c_fc1 = nn.Linear(output_dim, h_dim)
        self.c_fc2 = nn.Linear(h_dim, h_dim)
        self.c_fc3 = nn.Linear(h_dim, h_dim // 2)
        self.c_fc4 = nn.Linear(h_dim // 2, h_dim // 4)
        out_dim = h_dim // 4
        if self.is_self:
            self.c_fc5 = nn.Linear(h_dim / 4, h_dim // 8)
            out_dim = h_dim // 8

        self.c_logit = nn.Linear(out_dim, 1)
        self.c_tau = nn.Linear(out_dim, 1)
        
        # uplift net
        self.u_fc1 = nn.Linear(output_dim, h_dim)
        self.u_fc2 = nn.Linear(h_dim, h_dim)
        self.u_fc3 = nn.Linear(h_dim, h_dim // 2)
        self.u_fc4 = nn.Linear(h_dim // 2, h_dim // 4)
        out_dim = h_dim // 4
        if self.is_self:
            self.u_fc5 = nn.Linear(h_dim / 4, h_dim // 8)
            out_dim = h_dim // 8

        self.t_logit = nn.Linear(out_dim, 1)
        self.u_tau = nn.Linear(out_dim, 1)

        #self.d1 = nn.Parameter(data=torch.randn((1, 1), dtype=torch.float), requires_grad=True)  #做为模型的可训练参数
        self.temp = nn.Parameter(data=torch.ones((1, 1), dtype=torch.float), requires_grad=False)
        #self.delta = nn.Parameter(data=0.25 * torch.ones((1, 1), dtype=torch.float), requires_grad=False)

        # activation function
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError('unknown act_type {0}'.format(act_type))

    def forward(self, feature_list):
        # representation part
        #R_last = self.act(self.R_fc1(feature_list))

        # propensity for treat predict  治疗倾向预测
        # Propensity score主要是用来估计给定样本协变量情况下，被施加treatment的概率
        # t_fc1 = self.act(self.t_fc1(R_last))
        # t_logit = self.t_logit(t_fc1)
        R_last = self.moe(feature_list)
        # control net
        c_last = self.act(self.c_fc4(self.act(self.c_fc3(self.act(self.c_fc2(self.act(self.c_fc1(R_last))))))))
        if self.is_self:
            c_last = self.act(self.c_fc5(c_last))
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # uplift net
        u_last = self.act(self.u_fc4(self.act(self.u_fc3(self.act(self.u_fc2(self.act(self.u_fc1(R_last))))))))
        if self.is_self:
            u_last = self.act(self.u_fc5(u_last))
        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit) # 将 t_logit 转换为概率，用于最终的预测
        #建一个与 t_logit 具有相同的维度，且所有元素均为 self.d1 的张量 epsilons，这通常用于在后续计算中添加一个均匀的扰动或正则化项。
        #epsilons = self.d1 * torch.ones_like(t_logit)[:, 0:1]                                   

        return c_logit, c_prob, c_tau, t_logit, t_prob, u_tau

    def calculate_loss(self, feature_list, is_treat, label_list):
        # Model outputs
        c_logit, c_prob, c_tau, t_logit, t_prob, u_tau = self.forward(feature_list)

        # regression
        c_logit_fix = c_logit.detach() #创建一个新张量，与c_logit 共享相同的数据
        uc = c_logit
        ut = (c_logit_fix + u_tau)

        y_true = torch.unsqueeze(label_list, 1)
        t_true = torch.unsqueeze(is_treat, 1)

        '''Losses'''
        # response loss
        # 用均方误差损失
        temp = torch.square((1 - t_true) * uc + t_true * ut - y_true)
        #直接取平均值
        loss = torch.mean(temp)

        return loss
class M3TN1(nn.Module):
    """`
    M3TN: MULTI-GATE MIXTURE-OF-EXPERTS BASED MULTI-VALUED TREATMENT NETWORK FOR UPLIFT MODELING
    """
    def __init__(self, input_dim, h_dim, is_self, act_type):
        super(M3TN1, self).__init__()
        self.is_self = is_self
        #self.num_tasks = num_tasks
        # MoE layer (Mixture of Experts)
        #self.moe = MoE(input_dim, h_dim, output_dim, num_experts, dropout_rate=0.1)
        # representation part 引入MMoE
            # output_dim = 3
            # hidden_dim = 32
        
        # propensity for treat predict
        # self.t_fc1 = nn.Linear(h_dim, h_dim)
        # self.t_logit = nn.Linear(h_dim, 1)

        
        # control net
        self.c_fc1 = nn.Linear(input_dim, h_dim)
        self.c_fc2 = nn.Linear(h_dim, h_dim)
        self.c_fc3 = nn.Linear(h_dim, h_dim // 2)
        self.c_fc4 = nn.Linear(h_dim // 2, h_dim // 4)
        out_dim = h_dim // 4
        if self.is_self:
            self.c_fc5 = nn.Linear(h_dim / 4, h_dim // 8)
            out_dim = h_dim // 8

        self.c_logit = nn.Linear(out_dim, 1)
        self.c_tau = nn.Linear(out_dim, 1)
        
        # uplift net
        self.u_fc1 = nn.Linear(input_dim, h_dim)
        self.u_fc2 = nn.Linear(h_dim, h_dim)
        self.u_fc3 = nn.Linear(h_dim, h_dim // 2)
        self.u_fc4 = nn.Linear(h_dim // 2, h_dim // 4)
        out_dim = h_dim // 4
        if self.is_self:
            self.u_fc5 = nn.Linear(h_dim / 4, h_dim // 8)
            out_dim = h_dim // 8

        self.t_logit = nn.Linear(out_dim, 1)
        self.u_tau = nn.Linear(out_dim, 1)

        #self.d1 = nn.Parameter(data=torch.randn((1, 1), dtype=torch.float), requires_grad=True)  #做为模型的可训练参数
        self.temp = nn.Parameter(data=torch.ones((1, 1), dtype=torch.float), requires_grad=False)
        #self.delta = nn.Parameter(data=0.25 * torch.ones((1, 1), dtype=torch.float), requires_grad=False)

        # activation function
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError('unknown act_type {0}'.format(act_type))

    def forward(self, feature_list):
        # representation part
        #R_last = self.act(self.R_fc1(feature_list))

        # propensity for treat predict  治疗倾向预测
        # Propensity score主要是用来估计给定样本协变量情况下，被施加treatment的概率
        # t_fc1 = self.act(self.t_fc1(R_last))
        # t_logit = self.t_logit(t_fc1)
        #R_last = self.moe(feature_list)
        # control net
        c_last = self.act(self.c_fc4(self.act(self.c_fc3(self.act(self.c_fc2(self.act(self.c_fc1(feature_list))))))))
        if self.is_self:
            c_last = self.act(self.c_fc5(c_last))
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # uplift net
        u_last = self.act(self.u_fc4(self.act(self.u_fc3(self.act(self.u_fc2(self.act(self.u_fc1(feature_list))))))))
        if self.is_self:
            u_last = self.act(self.u_fc5(u_last))
        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit) # 将 t_logit 转换为概率，用于最终的预测
        #建一个与 t_logit 具有相同的维度，且所有元素均为 self.d1 的张量 epsilons，这通常用于在后续计算中添加一个均匀的扰动或正则化项。
        #epsilons = self.d1 * torch.ones_like(t_logit)[:, 0:1]                                   

        return c_logit, c_prob, c_tau, t_logit, t_prob, u_tau

    def calculate_loss(self, feature_list, is_treat, label_list):
        # Model outputs
        c_logit, c_prob, c_tau, t_logit, t_prob, u_tau = self.forward(feature_list)

        # regression
        c_logit_fix = c_logit.detach() #创建一个新张量，与c_logit 共享相同的数据
        uc = c_logit
        ut = (c_logit_fix + u_tau)

        y_true = torch.unsqueeze(label_list, 1)
        t_true = torch.unsqueeze(is_treat, 1)

        '''Losses'''
        # response loss
        # 用均方误差损失
        temp = torch.square((1 - t_true) * uc + t_true * ut - y_true)
        #直接取平均值
        loss = torch.mean(temp)

        return loss
class M3TN2(nn.Module):
    """`
    M3TN: MULTI-GATE MIXTURE-OF-EXPERTS BASED MULTI-VALUED TREATMENT NETWORK FOR UPLIFT MODELING
    """
    def __init__(self, input_dim, h_dim, output_dim, num_experts, is_self, act_type):
        super(M3TN2, self).__init__()
        self.is_self = is_self
        #self.num_tasks = num_tasks
        # MoE layer (Mixture of Experts)
        self.moe = MoE(input_dim, h_dim, output_dim, num_experts, dropout_rate=0.1)
        # representation part 引入MMoE
            # output_dim = 3
            # hidden_dim = 32
        
        # propensity for treat predict
        # self.t_fc1 = nn.Linear(h_dim, h_dim)
        # self.t_logit = nn.Linear(h_dim, 1)

        
        # control net
        self.s_fc1 = nn.Linear(output_dim+1, h_dim)
        self.s_fc2 = nn.Linear(h_dim, h_dim)
        self.s_fc3 = nn.Linear(h_dim, h_dim // 2)
        self.s_fc4 = nn.Linear(h_dim // 2, h_dim // 4)
        out_dim = h_dim // 4
        if self.is_self:
            self.c_fc5 = nn.Linear(h_dim / 4, h_dim // 8)
            out_dim = h_dim // 8

        self.s_logit = nn.Linear(out_dim, 1)

        # activation function
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError('unknown act_type {0}'.format(act_type))

    def forward(self, feature_list, is_treat):
        # representation part
        #R_last = self.act(self.R_fc1(feature_list))

        # propensity for treat predict  治疗倾向预测
        # Propensity score主要是用来估计给定样本协变量情况下，被施加treatment的概率
        # t_fc1 = self.act(self.t_fc1(R_last))
        # t_logit = self.t_logit(t_fc1)
        R_last = self.moe(feature_list)
        is_treat = torch.unsqueeze(is_treat, dim=1)
        xt = torch.cat((R_last, is_treat), dim=1)
        # 连接特征值和干预数据 dim=1按列拼接 为什么要这个拼接？
        s_last = self.act(self.s_fc4(self.act(self.s_fc3(self.act(self.s_fc2(self.act(self.s_fc1(xt))))))))
        #一层一层激活并使用相同的激活函数
        if self.is_self:
            s_last = self.act(self.s_fc5(s_last))                                                              #这个在干什么 #如果得到治疗需要多一步操作

        s_logit = self.s_logit(s_last)#输出层
                                                                                                               #是输出
        s_prob = torch.sigmoid(s_logit)   

        _xt = torch.cat((R_last, (1-is_treat)), dim=1)
        # 连接特征值和干预数据 dim=1按列拼接 为什么要这个拼接？
        _s_last = self.act(self.s_fc4(self.act(self.s_fc3(self.act(self.s_fc2(self.act(self.s_fc1(_xt))))))))
        #一层一层激活并使用相同的激活函数
        if self.is_self:
            _s_last = self.act(self.s_fc5(_s_last))                                                              #这个在干什么 #如果得到治疗需要多一步操作

        _s_logit = self.s_logit(_s_last)#输出层
                                                                                                               #是输出
        _s_prob = torch.sigmoid(_s_logit)  
        y0 = is_treat * _s_prob + (1 - is_treat) * s_prob  # t=0 取真实y0 t=1 取预测y0
        y1 = is_treat * s_prob + (1 - is_treat) * _s_prob                                                             #为什么分别乘以概率， *** 是考虑了反事实结果吗

        return s_logit, y1 - y0
    def calculate_loss(self, feature_list, is_treat, label_list):
            y_true = torch.unsqueeze(label_list, dim=1)
            #标签是一维的
            s_logit, _ = self.forward(feature_list, is_treat)
            #前向传播后形成的模型
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')# 损失函数 梯度下降应该在这里面
            #创建BCEWithLogitsLoss损失函数对象
            loss = criterion(s_logit, y_true)
            #计算损失
            return loss

        