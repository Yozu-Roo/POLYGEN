import math
import os
import random
import time

import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def stratified_sampling(molecules, x, t, ratio=0.75, bin_size=0.1):
    # 高分 & 低分的目标采样数量
    high_count = int(ratio * x)

    # 按 bin_size 进行分层
    high_bins = {}  # 存放高分样本 (score >= t)
    low_bins = {}  # 存放低分样本 (score < t)
    high_mols = []

    for m in molecules:
        bin_key = round(m.score // bin_size * bin_size, 1)  # 计算分层区间
        if m.score >= t:
            high_bins.setdefault(bin_key, []).append(m)
            high_mols.append(m)
        else:
            low_bins.setdefault(bin_key, []).append(m)

    def sample_from_bins(bins, sample_size):
        """从每个分层中按比例采样（不放回），如果某层不足则全采"""
        total_available = sum(len(b) for b in bins.values())  # 总可用样本数

        sampled = []
        for bin_key in sorted(bins.keys()):  # 按层从小到大采样
            bin_samples = bins[bin_key]
            bin_size = len(bin_samples)

            # 按比例分配采样数
            bin_target = int(sample_size * (bin_size / total_available))
            bin_target = min(bin_target, bin_size)  # 不能超出该层的实际数量

            sampled_now = random.sample(bin_samples, bin_target)
            sampled.extend(sampled_now)
            print(f"Layer of {bin_key} samples：{len(sampled_now)} / {bin_size}")

        return sampled

    # 采样高分 & 低分样本
    high_sampled = sample_from_bins(high_bins, high_count)

    low_count = len(high_sampled) * (1-ratio) / ratio
    low_sampled = sample_from_bins(low_bins, low_count)

    # 最终统计
    print(f"Samples of score >= {t}: {len(high_sampled)} \n"
          f"Samples of score < {t}: {len(low_sampled)}\n"
          f"total samples: {len(high_sampled) + len(low_sampled)}")
    # 返回最终采样的 SMILES 列表
    final_sample = high_sampled + low_sampled
    return high_mols, final_sample
