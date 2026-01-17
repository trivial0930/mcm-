#读取文件csv/tsv格式
import pandas as pd
import os

def load_data(file_path):
    """
    通用数据读取函数
    功能：
    1. 自动识别 .csv (逗号分隔) 和 .tsv (制表符分隔)
    2. 自动处理编码问题 (UTF-8 vs Latin1)
    
    参数:
        file_path (str): 文件的路径，例如 'data/hair_dryer.tsv'
        
    返回:
        df (DataFrame): 读取成功的Pandas数据框，如果失败返回 None
    """
    
    # --- 第一步：判断文件类型 ---
    # os.path.splitext 会把文件名分成 ('文件名', '.后缀')
    # 比如 'data.tsv' -> ( 'data', '.tsv' )
    _, file_extension = os.path.splitext(file_path)
    
    # 根据后缀决定分隔符 (separator)
    # TSV 用 \t (Tab键) 分隔，CSV 用 , (逗号) 分隔
    if file_extension.lower() == '.tsv':
        sep = '\t'
    else:
        sep = ','  # 默认为 csv

    print(f"正在读取文件: {file_path} (类型: {file_extension}, 分隔符: '{sep}')")

    # --- 第二步：尝试读取 (处理编码乱码) ---
    df = None
    
    # 尝试方案 A: 使用标准的 utf-8 编码 (绝大多数现代数据)
    try:
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
        print("✅ 成功使用 UTF-8 编码读取")
        
    except UnicodeDecodeError:
        # 尝试方案 B: 如果 utf-8 报错，尝试 latin1 (常见于欧美老旧系统数据)
        print("⚠️ UTF-8 读取失败，正在尝试 Latin1 编码...")
        try:
            df = pd.read_csv(file_path, sep=sep, encoding='latin1')
            print("✅ 成功使用 Latin1 编码读取")
        except Exception as e:
            print(f"❌ 读取彻底失败，请检查文件损坏或格式问题。错误信息: {e}")
            return None
            
    # --- 第三步：返回数据 ---
    print(f"数据形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
    return df

# --- 测试代码 (仅当直接运行此文件时执行) ---
if __name__ == "__main__":
    file_path = r"D:\download\mcm2020c\Problem_C_Data\Problem_C_Data\hair_dryer.tsv"

    if os.path.exists(file_path):
        print("📂 文件路径存在，准备读取...")
        df = load_data(file_path)
        
        if df is not None:
            print("\n--- 读取成功！数据预览 (前5行) ---")
            print(f"数据维度: {df.shape}")  # 先打印形状，确认读进来了多少行
            print(f"列名列表: {df.columns.tolist()}") # 确认列名没乱码
            pd.set_option('display.max_colwidth', 50) 
            # 强制不换行显示，方便横向查看
            pd.set_option('display.expand_frame_repr', False) 
            
            print("\n--- 数据预览 (前5行) ---")
            print(df.head())
    else:
        print(f"❌ 错误：找不到文件。请检查路径是否正确：\n{file_path}")

    
