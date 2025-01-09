import torch
import numpy as np
import pandas as pd


def uplift_at_k(y_true, uplift, treatment, strategy, k=0.3):
    n_samples = len(y_true)
    order = np.argsort(uplift, kind='mergesort')[::-1]#-1是降序排序 kind是排序规则

    if strategy == 'overall': # 全部数据
        n_size = int(n_samples * k)

        # ToDo: _checker_ there are observations among two groups among first k
        score_ctrl = y_true[order][:n_size][treatment[order][:n_size] == 0].mean() # 计算对照组的平均得分
        score_trmnt = y_true[order][:n_size][treatment[order][:n_size] == 1].mean() # 计算处理组的平均得分
        print(score_ctrl)
        print(score_trmnt)
        if np.isnan(score_ctrl):#判断是否是空值
            score_ctrl = 0
        if np.isnan(score_trmnt):
            score_trmnt = 0

    else:  # strategy == 'by_group': # 采样数据
        n_ctrl = int((treatment == 0).sum() * k)
        n_trmnt = int((treatment == 1).sum() * k)

        score_ctrl = y_true[order][treatment[order] == 0][:n_ctrl].mean()
        score_trmnt = y_true[order][treatment[order] == 1][:n_trmnt].mean()
        print(score_ctrl)
        print(score_trmnt)
        if np.isnan(score_ctrl):
            score_ctrl = 0
        if np.isnan(score_trmnt):
            score_trmnt = 0

    return score_trmnt - score_ctrl


def response_rate_by_percentile(y_true, uplift, treatment, group, strategy='overall', bins=10):
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment) #类似数组
    order = np.argsort(uplift, kind='mergesort')[::-1]

    trmnt_flag = 1 if group == 'treatment' else 0 #处理组为0控制组为1

    if strategy == 'overall':
        y_true_bin = np.array_split(y_true[order], bins)#可不均等分为10个部分 真实值
        trmnt_bin = np.array_split(treatment[order], bins) # 处理组
        #10个部分分别去计算
        group_size = np.array([len(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)])#并行遍历 计算每个子数组中符合条件的y的个数
        response_rate = np.array([np.mean(y[trmnt == trmnt_flag]) for y, trmnt in zip(y_true_bin, trmnt_bin)]) #计算每个子数组中符合条件的y的响应率（平均值）

    else:  # strategy == 'by_group'
        y_bin = np.array_split(y_true[order][treatment[order] == trmnt_flag], bins)

        group_size = np.array([len(y) for y in y_bin])
        response_rate = np.array([np.mean(y) for y in y_bin])

    response_rate[np.isnan(response_rate)] = 0 # 将 response_rate 中的 NaN 值替换为 0，避免出现错误
    _group_size = group_size.copy()
    _group_size[_group_size == 0] = 1
    variance = np.multiply(response_rate, np.divide((1 - response_rate), _group_size)) # 计算每个组的方差

    return response_rate, variance, group_size


def weighted_average_uplift(y_true, uplift, treatment, strategy='overall', bins=10):
    response_rate_trmnt, variance_trmnt, n_trmnt = response_rate_by_percentile(
        y_true, uplift, treatment, group='treatment', strategy=strategy, bins=bins)

    response_rate_ctrl, variance_ctrl, n_ctrl = response_rate_by_percentile(
        y_true, uplift, treatment, group='control', strategy=strategy, bins=bins)

    uplift_scores = response_rate_trmnt - response_rate_ctrl

    weighted_avg_uplift = np.dot(n_trmnt, uplift_scores) / np.sum(n_trmnt) #dot是向量点积 计算加权平均提升分数

    return weighted_avg_uplift


def uplift_by_percentile(y_true, uplift, treatment, strategy='overall',
                         bins=10, std=False, total=False, string_percentiles=True): # 用于进一步可视化分析
    response_rate_trmnt, variance_trmnt, n_trmnt = response_rate_by_percentile(
        y_true, uplift, treatment, group='treatment', strategy=strategy, bins=bins)

    response_rate_ctrl, variance_ctrl, n_ctrl = response_rate_by_percentile(
        y_true, uplift, treatment, group='control', strategy=strategy, bins=bins)

    uplift_scores = response_rate_trmnt - response_rate_ctrl #提升分数
    uplift_variance = variance_trmnt + variance_ctrl #提升方差

    percentiles = [round(p * 100 / bins) for p in range(1, bins + 1)] #这意味着将数据划分为 10 个部分，分别表示前 10%、20%、30%...到 100% 的数据分布

    if string_percentiles:
        percentiles = [f"0-{percentiles[0]}"] + \
            [f"{percentiles[i]}-{percentiles[i + 1]}" for i in range(len(percentiles) - 1)] # + \这是将前面创建的列表与后面生成的列表连接起来

    df = pd.DataFrame({ # 创建一个 Pandas 数据框 df
        'percentile': percentiles,
        'n_treatment': n_trmnt,
        'n_control': n_ctrl,
        'response_rate_treatment': response_rate_trmnt,
        'response_rate_control': response_rate_ctrl,
        'uplift': uplift_scores
    })

    if total: # 将总体统计数据插入到数据框的第一行，方便后续分析
        response_rate_trmnt_total, variance_trmnt_total, n_trmnt_total = response_rate_by_percentile(
            y_true, uplift, treatment, strategy=strategy, group='treatment', bins=1)

        response_rate_ctrl_total, variance_ctrl_total, n_ctrl_total = response_rate_by_percentile(
            y_true, uplift, treatment, strategy=strategy, group='control', bins=1)

        df.loc[-1, :] = ['total', n_trmnt_total, n_ctrl_total, response_rate_trmnt_total,
                         response_rate_ctrl_total, response_rate_trmnt_total - response_rate_ctrl_total]

    if std: # 计算标准差
        std_treatment = np.sqrt(variance_trmnt)
        std_control = np.sqrt(variance_ctrl)
        std_uplift = np.sqrt(uplift_variance)

        if total:
            std_treatment = np.append(std_treatment, np.sum(std_treatment))
            std_control = np.append(std_control, np.sum(std_control))
            std_uplift = np.append(std_uplift, np.sum(std_uplift))
        # 将计算得到的标准差添加到数据框 df 中的新列 
        df.loc[:, 'std_treatment'] = std_treatment
        df.loc[:, 'std_control'] = std_control
        df.loc[:, 'std_uplift'] = std_uplift

    df = df \
        .set_index('percentile', drop=True, inplace=False) \
        .astype({'n_treatment': 'int32', 'n_control': 'int32'})

    return df


def euen_optimizers(model, weight_decay, learning_rate):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # 根据是否应用权重衰减将模型参数分为两组。这种分类有助于提高模型的训练效果，并减少过拟合的风险。
    # 通过对偏置和某些层（如 LayerNorm 和 Embedding）的权重不施加权重衰减，该函数实现了更为灵活的正则化策略。
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set() #集合
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,) # 只有这些模块的权重会进行权重衰减 全连接层
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding) # 这些模块的权重不会进行权重衰减 归一化层、嵌入层
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name 构造每个参数的完整名称

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed 白名单
                if fpn.startswith('c_fc1') or fpn.startswith('u_fc1'):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed 黑名单
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay #检查是否有参数被错误地分类到两个集合中
    union_params = decay | no_decay #确保所有参数都被分类
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def tarnet_optimizers(model, weight_decay, learning_rate, step=1):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            if pn.endswith('bias') or pn.endswith('d1'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                if fpn.startswith('D_fc1') or fpn.startswith('D_fc2') or fpn.startswith('t_logit'):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    if step == 1:
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=0.9, nesterov=True)
    return optimizer


def cfrnet_optimizers(model, weight_decay, learning_rate):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                if fpn.startswith('B_ifc1') or fpn.startswith('B_ifc2'):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def dragonnet_optimizers(model, weight_decay, learning_rate, step=1):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            if pn.endswith('bias') or pn.endswith('d1'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                if fpn.startswith('D_fc1') or fpn.startswith('D_fc2') or fpn.startswith('t_logit'):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    if step == 1:
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=0.9, nesterov=True)
    return optimizer
