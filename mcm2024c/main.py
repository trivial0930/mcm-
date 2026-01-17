# main.py
import pandas as pd

from data_loader import load_data          # 你们的加载模块 :contentReference[oaicite:3]{index=3}
from data_clean import DataCleaner         # 你们的清洗模块 :contentReference[oaicite:4]{index=4}
from visual_utils import set_mcm_style      # 你们的可视化风格 :contentReference[oaicite:5]{index=5}

from momentum_train import train_momentum_model


DATA_DICT_PATH = r"D:\download\mcm2024c\data_dictionary.csv"
MATCH_PATH = r"D:\download\mcm2024c\Wimbledon_featured_matches.csv"


def main():
    set_mcm_style("seaborn")

    # 1) Load
    df_dict = load_data(DATA_DICT_PATH)
    df = load_data(MATCH_PATH)

    # 2) Clean (按你们风格来)
    cleaner = DataCleaner(df)
    cleaner.inspect_data()
    df = cleaner.remove_duplicates()

    # （可选）缺失值策略示例：按你们数据实际缺失情况调整
    # df = cleaner.handle_missing({
    #     "speed_mph": "median",
    #     "rally_count": "median",
    # })

    # 3) Train momentum model
    best_params, p_value, df_out = train_momentum_model(
    df,
    objective="future_winrate",
    K=5,
    weight_per_point=True,
)


    print("\n===== Best Momentum Params =====")
    for k, v in best_params.items():
        print(f"{k:>12} : {v}")
    print(f"\nLRT p-value: {p_value:.6g}")

    # 4) Save output
    out_path = r"D:\download\mcm2024c\with_momentum.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()

