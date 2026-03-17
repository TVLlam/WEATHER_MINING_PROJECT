"""
Module: AssociationMiner
Khai phá Luật Kết hợp (Association Rules) trên dữ liệu thời tiết.

ĐIỂM CỐT LÕI: Tách dataframe theo từng mùa (Season) và chạy
thuật toán FP-Growth/Apriori riêng cho từng mùa → So sánh luật giữa các mùa.
"""

import os
import logging
from typing import Optional

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.data.loader import load_config, resolve_path

logger = logging.getLogger(__name__)


class AssociationMiner:
    """
    Khai phá luật kết hợp từ dữ liệu thời tiết đã rời rạc hóa.

    Parameters
    ----------
    config : dict, optional
        Dictionary config từ params.yaml.
    """

    # Các cột rời rạc dùng cho association mining
    ITEM_COLS = ["Precip Type", "Summary", "Temp_Bin", "Humidity_Bin", "Wind_Bin"]

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.assoc_cfg = self.config["association"]
        self.min_support = self.assoc_cfg["min_support"]
        self.min_confidence = self.assoc_cfg["min_confidence"]
        self.min_lift = self.assoc_cfg["min_lift"]
        self.top_n = self.assoc_cfg["top_n_rules"]
        self.algorithm = self.assoc_cfg.get("algorithm", "fpgrowth")
        self.rules_by_season = {}
        self.all_rules = None

    def _prepare_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi DataFrame thành dạng one-hot encoding cho mlxtend.

        Mỗi hàng là một 'transaction' chứa các item dạng 'Column=Value',
        ví dụ: 'Temp_Bin=High', 'Humidity_Bin=Low', 'Summary=Clear'.
        """
        # Tạo danh sách transactions
        item_cols = [c for c in self.ITEM_COLS if c in df.columns]
        transactions = []
        for _, row in df[item_cols].iterrows():
            transaction = [f"{col}={row[col]}" for col in item_cols if pd.notna(row[col])]
            transactions.append(transaction)

        # Transaction Encoding → one-hot DataFrame
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        return df_encoded

    def _mine_rules(self, df_encoded: pd.DataFrame, label: str = "") -> pd.DataFrame:
        """
        Chạy thuật toán tìm frequent itemsets và sinh luật kết hợp.

        Parameters
        ----------
        df_encoded : pd.DataFrame
            DataFrame đã one-hot encoding.
        label : str
            Nhãn mùa (để log).

        Returns
        -------
        pd.DataFrame
            DataFrame chứa các luật kết hợp.
        """
        prefix = f"[{label}] " if label else ""

        # Tìm frequent itemsets
        if self.algorithm == "fpgrowth":
            freq_items = fpgrowth(df_encoded, min_support=self.min_support, use_colnames=True)
        else:
            freq_items = apriori(df_encoded, min_support=self.min_support, use_colnames=True)

        if freq_items.empty:
            logger.warning("%sKhông tìm thấy frequent itemsets (min_support=%.3f)", prefix, self.min_support)
            return pd.DataFrame()

        logger.info("%sTìm được %d frequent itemsets", prefix, len(freq_items))

        # Sinh luật kết hợp
        rules = association_rules(freq_items, metric="confidence", min_threshold=self.min_confidence)

        if rules.empty:
            logger.warning("%sKhông có luật nào thỏa min_confidence=%.2f", prefix, self.min_confidence)
            return pd.DataFrame()

        # Lọc theo Lift
        rules = rules[rules["lift"] >= self.min_lift].copy()

        # Chuyển frozenset thành string dễ đọc
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        rules["rule"] = rules["antecedents_str"] + " → " + rules["consequents_str"]

        # Sort theo Lift giảm dần, lấy Top N
        rules = rules.sort_values("lift", ascending=False).head(self.top_n)

        logger.info("%sTop %d luật (Lift ≥ %.1f, Confidence ≥ %.2f):",
                     prefix, len(rules), self.min_lift, self.min_confidence)
        for _, r in rules.head(5).iterrows():
            logger.info("  %s  [Conf=%.2f, Lift=%.2f, Supp=%.3f]",
                        r["rule"], r["confidence"], r["lift"], r["support"])

        return rules

    def mine_rules_by_season(self, df: pd.DataFrame) -> dict:
        """
        ĐIỂM CỐT LÕI: Tách dataframe theo từng mùa, chạy khai phá riêng.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame chứa cột 'Season' và các cột rời rạc.

        Returns
        -------
        dict
            {season_name: rules_dataframe}
        """
        logger.info("=" * 60)
        logger.info("KHAI PHÁ LUẬT KẾT HỢP THEO TỪNG MÙA")
        logger.info("=" * 60)

        if "Season" not in df.columns:
            raise ValueError("DataFrame thiếu cột 'Season'. Hãy chạy FeatureBuilder trước.")

        seasons = df["Season"].unique()
        self.rules_by_season = {}

        for season in sorted(seasons):
            logger.info("\n--- Mùa: %s ---", season)
            df_season = df[df["Season"] == season]
            logger.info("Số mẫu: %d", len(df_season))

            df_encoded = self._prepare_transactions(df_season)
            rules = self._mine_rules(df_encoded, label=season)

            if not rules.empty:
                rules["Season"] = season

            self.rules_by_season[season] = rules

        return self.rules_by_season

    def compare_seasons(self) -> pd.DataFrame:
        """
        So sánh các luật kết hợp giữa các mùa.

        Returns
        -------
        pd.DataFrame
            DataFrame tổng hợp luật từ tất cả các mùa, sắp xếp theo Lift.
        """
        all_dfs = []
        for season, rules in self.rules_by_season.items():
            if not rules.empty:
                subset = rules[["Season", "rule", "support", "confidence", "lift"]].copy()
                all_dfs.append(subset)

        if not all_dfs:
            logger.warning("Không có luật nào để so sánh!")
            return pd.DataFrame()

        self.all_rules = pd.concat(all_dfs, ignore_index=True)
        self.all_rules = self.all_rules.sort_values(["Season", "lift"], ascending=[True, False])

        logger.info("\nTổng hợp %d luật từ %d mùa", len(self.all_rules), len(self.rules_by_season))
        return self.all_rules

    def interpret_rules(self) -> list:
        """
        Tự động diễn giải các luật kết hợp nổi bật theo từng mùa.

        Returns
        -------
        list of str
            Danh sách các diễn giải bằng ngôn ngữ tự nhiên.
        """
        interpretations = []

        for season, rules in self.rules_by_season.items():
            if rules.empty:
                interpretations.append(f"Mùa {season}: Không tìm thấy luật nổi bật.")
                continue

            top_rule = rules.iloc[0]
            interp = (
                f"🔹 Mùa {season}: Nếu {top_rule['antecedents_str']}, "
                f"thì {top_rule['consequents_str']} "
                f"(Confidence={top_rule['confidence']:.1%}, "
                f"Lift={top_rule['lift']:.2f})"
            )
            interpretations.append(interp)

            # Thêm diễn giải cho top 3 luật
            for i, (_, r) in enumerate(rules.head(3).iterrows()):
                detail = (
                    f"   Luật {i+1}: {r['rule']} "
                    f"[Support={r['support']:.3f}, "
                    f"Confidence={r['confidence']:.1%}, "
                    f"Lift={r['lift']:.2f}]"
                )
                interpretations.append(detail)

        return interpretations

    def run(self, df: pd.DataFrame) -> dict:
        """
        Chạy toàn bộ pipeline khai phá luật kết hợp.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame đã qua Feature Engineering.

        Returns
        -------
        dict
            Kết quả gồm rules_by_season, comparison, interpretations.
        """
        # 1. Khai phá theo từng mùa
        self.mine_rules_by_season(df)

        # 2. So sánh giữa các mùa
        comparison = self.compare_seasons()

        # 3. Diễn giải tự động
        interpretations = self.interpret_rules()

        logger.info("\n" + "=" * 60)
        logger.info("DIỄN GIẢI LUẬT KẾT HỢP:")
        logger.info("=" * 60)
        for line in interpretations:
            logger.info(line)

        return {
            "rules_by_season": self.rules_by_season,
            "comparison": comparison,
            "interpretations": interpretations,
        }

    def save_results(self, output_dir: str = None) -> str:
        """
        Lưu kết quả ra file CSV.

        Parameters
        ----------
        output_dir : str, optional
            Thư mục đầu ra. Mặc định dùng outputs/tables/.

        Returns
        -------
        str
            Đường dẫn file đã lưu.
        """
        if output_dir is None:
            output_dir = resolve_path(self.config["outputs"]["tables_dir"])
        os.makedirs(output_dir, exist_ok=True)

        if self.all_rules is not None and not self.all_rules.empty:
            save_path = os.path.join(output_dir, "association_rules_by_season.csv")
            self.all_rules.to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info("Đã lưu luật kết hợp tại: %s", save_path)
            return save_path

        logger.warning("Không có kết quả để lưu.")
        return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    from src.data.loader import load_config, resolve_path
    config = load_config()
    parquet_path = resolve_path(config["data"]["cleaned_parquet"])
    df = pd.read_parquet(parquet_path)

    miner = AssociationMiner(config)
    results = miner.run(df)
    miner.save_results()

    print("\n✅ Association Mining hoàn tất!")
