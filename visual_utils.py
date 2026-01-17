import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 全局样式设置 (Global Style Settings)
# ==========================================
def set_mcm_style(style='seaborn'):
    """
    设置美赛绘图风格。
    :param style: 'science' (需安装LaTeX), 'seaborn' (默认), 'dark'
    """
    # 基础配置：字体与负号显示
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    if style == 'science':
        try:
            import scienceplots
            plt.style.use(['science', 'ieee']) # IEEE风格，非常适合论文
        except ImportError:
            print("Warning: scienceplots not found. Using seaborn style.")
            sns.set_theme(style="whitegrid", font='Times New Roman')
    elif style == 'seaborn':
        sns.set_theme(style="whitegrid", rc={"axes.facecolor": ".98"})
        sns.set_context("paper", font_scale=1.2) # 论文语境，字体稍大
    
    print(f"Style set to: {style}")

def save_fig(fig, filename, output_dir='figures'):
    """保存图片为高清格式，自动处理路径"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    path = os.path.join(output_dir, filename)
    # bbox_inches='tight' 去除多余白边
    fig.savefig(path, dpi=300, bbox_inches='tight') 
    print(f"Figure saved to {path}")

# ==========================================
# 2. 高级绘图函数 (Advanced Plotting)
# ==========================================

def plot_heatmap(df, title="Correlation Heatmap", output_name="heatmap.png"):
    """
    绘制相关性热力图 (C题/数据挖掘必备)
    用于特征筛选，展示变量间的线性关系。
    """
    plt.figure(figsize=(10, 8))
    
    # 计算相关系数
    corr = df.corr()
    
    # 生成上三角掩码 (对角线以上不显示，减少冗余)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 绘制
    # cmap推荐: 'vlag' (冷暖色), 'coolwarm', 'RdBu_r'
    sns.heatmap(corr, mask=mask, cmap='vlag', vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, fmt=".2f", annot_kws={"size": 8})
    
    plt.title(title, fontsize=14, pad=20)
    save_fig(plt.gcf(), output_name)
    plt.show()

def plot_pairplot(df, hue=None, output_name="pairplot.png"):
    """
    绘制散点矩阵图 (C题/多变量分析)
    用于观察多变量两两之间的分布关系及分类边界。
    """
    # pairplot 返回的是 Grid 对象，不是 figure
    g = sns.pairplot(df, hue=hue, corner=True, palette='deep', 
                     plot_kws={'alpha': 0.6, 's': 30},
                     diag_kws={'fill': True})
    
    g.fig.suptitle(f"Pairplot of Variables (Hue: {hue})", y=1.02, fontsize=14)
    save_fig(g.fig, output_name)
    plt.show()

def plot_violin(df, x_col, y_col, hue=None, title="Violin Plot", output_name="violin.png"):
    """
    绘制小提琴图 (数据分布分析)
    比箱线图更高级，能看到数据的核密度分布。
    """
    plt.figure(figsize=(10, 6))
    
    sns.violinplot(data=df, x=x_col, y=y_col, hue=hue,
                   split=True, inner="quart", palette="muted")
    
    plt.title(title, fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    save_fig(plt.gcf(), output_name)
    plt.show()

def plot_dual_axis(x, y1, y2, label1="Y1", label2="Y2", title="Dual Axis Plot", output_name="dual_axis.png"):
    """
    绘制双Y轴图表 (D题/动态系统/时间序列)
    用于量纲不同的两个变量随时间的变化 (例如：人口 vs 增长率)。
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制左轴
    color1 = 'tab:blue'
    ax1.set_xlabel('Time/Step', fontsize=12)
    ax1.set_ylabel(label1, color=color1, fontsize=12)
    line1 = ax1.plot(x, y1, color=color1, label=label1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 绘制右轴
    ax2 = ax1.twinx()  # 实例化共享x轴的第二个轴
    color2 = 'tab:red'
    ax2.set_ylabel(label2, color=color2, fontsize=12)
    line2 = ax2.plot(x, y2, color=color2, label=label2, linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3) # 仅显示主轴网格
    save_fig(fig, output_name)
    plt.show()