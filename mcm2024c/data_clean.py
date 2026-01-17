# data_clean.py
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        """
        初始化清洗器
        :param df: 原始 DataFrame
        """
        self.df = df.copy()
        print(f"🔧 DataCleaner 已初始化，初始数据维度: {self.df.shape}")

    def inspect_data(self):
        """
        打印数据概览：缺失值统计、重复值统计
        """
        print("\n--- [数据体检报告] ---")
        print(f"总行数: {len(self.df)}")
        print(f"重复行数: {self.df.duplicated().sum()}")
        
        # 统计每一列的缺失值数量
        missing = self.df.isnull().sum()
        # 只打印有缺失值的列
        print("缺失值统计 (仅显示存在的列):")
        if missing[missing > 0].empty:
            print("  无缺失值")
        else:
            print(missing[missing > 0])
        print("----------------------")

    def clean_datetime(self, col_name):
        """
        修复警告: 指定 format='%H:%M:%S' 让 pandas 精确解析时间
        """
        if col_name in self.df.columns:
            print(f"正在转换日期列: {col_name}...")
            # 针对温网数据 '00:00:00' 这种格式，指定 format
            try:
                self.df[col_name] = pd.to_datetime(self.df[col_name], format='%H:%M:%S', errors='coerce')
            except Exception as e:
                # 如果指定格式失败，再尝试自动推断（兜底方案）
                print(f"⚠️ 指定格式解析失败，尝试自动推断: {e}")
                self.df[col_name] = pd.to_datetime(self.df[col_name], errors='coerce')
        else:
            print(f"⚠️ 列 {col_name} 不存在，跳过日期转换。")
        return self.df

    def remove_duplicates(self):
        """
        去除重复数据
        """
        original_count = len(self.df)
        self.df = self.df.drop_duplicates()
        new_count = len(self.df)
        if original_count != new_count:
            print(f"✂️ 已删除重复行: {original_count - new_count} 行")
        return self.df

    def handle_missing(self, fill_map=None):
        """
        🔥 修复的核心: 独立定义处理缺失值的函数
        :param fill_map: 字典，格式如 {'col_name': 'mean', 'col_name2': 0}
        """
        if not fill_map:
            return self.df

        print(f"正在处理缺失值，策略: {fill_map}")
        
        for col, strategy in fill_map.items():
            if col not in self.df.columns:
                continue
            
            # 记录处理前的空值数
            nan_count = self.df[col].isnull().sum()
            if nan_count == 0:
                continue

            if strategy == 'mean':
                # 只有数值型才能求均值
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    val = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(val)
                    print(f"  -> {col}: 填充均值 ({val:.2f})")
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(val)
                    print(f"  -> {col}: 填充中位数 ({val:.2f})")
            else:
                # 固定值填充 (如 0, 'Unknown')
                self.df[col] = self.df[col].fillna(strategy)
                print(f"  -> {col}: 填充固定值 ({strategy})")
                
        return self.df
