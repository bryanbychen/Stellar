import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 课程列表：每项为 (课程名, 学分, 百分制成绩, 是否预测课)。已修/已出分为 False，预估或在读为 True。
COURSES_DATA = [
    ("思想道德与法治", 2.5, 85, False),
    ("军事理论", 2.0, 91, False),
    ("大学生心理调适与发展", 1.5, 91, False),
    ("军事技能", 2.0, 87, False),
    ("形势与政策(一)", 1.0, 95, False),
    ("大学英语(一)", 3.0, 88, False),
    ("高等数学B(一)", 5.0, 70, False),
    ("计算机科学导论", 2.0, 97, False),
    ("线性代数", 3.0, 81, False),
    ("C语言程序设计", 4.0, 86, False),
    ("中国近现代史纲要", 2.5, 90, False),
    ("大学体育—游泳", 1.0, 90, False),
    ("大学物理实验B", 1.0, 70, False),
    ("C语言课程设计", 1.0, 93, False),
    ("大学生职业生涯规划(一)", 0.5, 97, False),
    ("大学英语(二)", 3.0, 85, False),
    ("大学物理B", 5.0, 65, False),
    ("高等数学B(二)", 5.0, 79, False),
    ("数据结构", 5.0, 71, False),
    ("数字电路", 3.5, 82, False),
    ("python", 2.5, 88, False),
    ("多媒体", 2.5, 95, False),
    ("体育", 1.0, 95, False),
    ("离散数学", 4.0, 95, False),
    ("党史", 0.5, 96, False),
    ("大职规", 0.5, 97, False),
    ("新技术讲座", 1, 89, False),
    ("马克思主义原理", 3, 88, False),
    ("教材研究", 1.0, 81, False),
    ("计算机组成原理", 3.5, 96, False),
    ("班级经营", 1.5, 78, False),
    ("概率与数理统计", 3, 99, False),
    ("数据库原理应用", 3.5, 90, False),
    ("数据结构实验", 1, 86, False),
    ("思想政治理论课社会实践", 2, 97, False),
    ("预测-音乐鉴赏", 1, 95, True),
    ("预测-体育", 1, 95, True),
    ("预测-开源硬件概论", 2.5, 95, True),
    ("预测-计算机网络", 3.5, 95, True),
    ("预测-发展与教育心理学", 2.0, 90, True),
    ("预测-计算机科学与技术教学理论与实践", 2.0, 85, True),
    ("预测-操作系统", 3.0, 95, True),
    ("预测-教育学", 1.5, 90, True),
    ("预测-金融量化分析与机器学习(Python)", 2.5, 95, True),
    ("预测-人工智能基础", 2.5, 95, True),
    ("预测-毛泽东思想和中国特色社会主义理论体系概论", 2.5, 90, True),
    ("预测-习近平新时代中国特色社会主义思想概论", 2.5, 90, True),
    ("预测-大学外语(三)", 2.0, 95, True),
    ("预测-劳动教育概论", 0.5, 95, True),
]


def prompt_float(label, default=None):
    """提示输入浮点数，回车可使用默认值。"""
    hint = f" [直接回车默认 {default}]" if default is not None else ""
    while True:
        raw = input(f"{label}{hint}: ").strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("  请输入有效数字（可含小数点）。")


class ZJNUGPACalculator:
    def __init__(self, target_gpa, total_graduation_credits):
        self.target_gpa = target_gpa
        self.total_graduation_credits = total_graduation_credits
        self.courses = []

    def add_course(self, name, credit, score, is_predicted=False):
        self.courses.append(
            {
                "course_name": name,
                "credit": float(credit),
                "score": float(score),
                "is_predicted": is_predicted,
            }
        )

    def _score_to_gpa(self, score):
        s = float(score)
        if s >= 100:
            return 5.0
        elif 95 <= s <= 99.9:
            return 4.5
        elif 90 <= s <= 94.9:
            return 4.0
        elif 85 <= s <= 89.9:
            return 3.5
        elif 80 <= s <= 84.9:
            return 3.0
        elif 75 <= s <= 79.9:
            return 2.5
        elif 70 <= s <= 74.9:
            return 2.0
        elif 65 <= s <= 69.9:
            return 1.5
        elif 60 <= s <= 64.9:
            return 1.0
        else:
            return 0.0

    def generate_report(self):
        df = pd.DataFrame(self.courses)
        df["gpa_point"] = df["score"].apply(self._score_to_gpa)
        df["weighted_gpa"] = df["gpa_point"] * df["credit"]

        df["weighted_score"] = df["score"] * df["credit"]
        arithmetic_avg = df["score"].mean()
        weighted_avg = df["weighted_score"].sum() / df["credit"].sum()

        recorded_credits = df["credit"].sum()
        total_recorded_points = df["weighted_gpa"].sum()

        df_real = df[df["is_predicted"] == False]
        real_credits = df_real["credit"].sum()
        real_gpa = (
            (df_real["gpa_point"] * df_real["credit"]).sum() / real_credits
            if real_credits > 0
            else 0
        )
        real_arithmetic_avg = df_real["score"].mean() if real_credits > 0 else 0
        real_weighted_avg = (
            (df_real["score"] * df_real["credit"]).sum() / real_credits
            if real_credits > 0
            else 0
        )

        predicted_mask = df["is_predicted"] == True
        df_90 = df.copy()
        df_85 = df.copy()
        df_90.loc[predicted_mask, "score"] = 90
        df_85.loc[predicted_mask, "score"] = 85
        df_90["gpa_point"] = df_90["score"].apply(self._score_to_gpa)
        df_90["weighted_gpa"] = df_90["gpa_point"] * df_90["credit"]
        df_85["gpa_point"] = df_85["score"].apply(self._score_to_gpa)
        df_85["weighted_gpa"] = df_85["gpa_point"] * df_85["credit"]
        gpa_90 = df_90["weighted_gpa"].sum() / df_90["credit"].sum()
        gpa_85 = df_85["weighted_gpa"].sum() / df_85["credit"].sum()
        avg_score_90 = (df_90["score"] * df_90["credit"]).sum() / df_90["credit"].sum()
        avg_score_85 = (df_85["score"] * df_85["credit"]).sum() / df_85["credit"].sum()
        arith_90 = df_90["score"].mean()
        arith_85 = df_85["score"].mean()

        remaining_unknown_credits = self.total_graduation_credits - recorded_credits
        planned_gpa = total_recorded_points / recorded_credits

        print("=" * 60)
        print(f"[报告] ZJNU 战略分析报告 | 毕业总学分设定: {self.total_graduation_credits}")
        print("=" * 60)
        print(
            f"【已规划】总学分 (已修+预测): {recorded_credits:.1f} | 毕业要求总学分: {self.total_graduation_credits:.1f}"
        )
        print(
            f"【已修】GPA: {real_gpa:.4f} | 算术平均: {real_arithmetic_avg:.2f} | 加权平均: {real_weighted_avg:.2f}"
        )
        print(
            f"【所有课程】GPA: {planned_gpa:.4f} | 算术平均: {arithmetic_avg:.2f} | 加权平均: {weighted_avg:.2f}"
        )
        print("-" * 30)
        print("【区间分析】预测课若全按 90 分估算:")
        print("GPA: %.4f | 算术平均: %.2f | 加权平均: %.2f" % (gpa_90, arith_90, avg_score_90))
        print("【区间分析】预测课若全按 85 分估算:")
        print("GPA: %.4f | 算术平均: %.2f | 加权平均: %.2f" % (gpa_85, arith_85, avg_score_85))
        print("-" * 30)

        resit_courses = df_real[df_real["score"] <= 79].copy()
        if not resit_courses.empty:
            df_resit = df.copy()
            for idx in resit_courses.index:
                df_resit.loc[idx, "score"] = 90
            df_resit["gpa_point"] = df_resit["score"].apply(self._score_to_gpa)
            df_resit["weighted_gpa"] = df_resit["gpa_point"] * df_resit["credit"]
            df_resit["weighted_score"] = df_resit["score"] * df_resit["credit"]
            credits_r = df_resit["credit"].sum()
            gpa_r = df_resit["weighted_gpa"].sum() / credits_r
            arith_r = df_resit["score"].mean()
            weight_r = df_resit["weighted_score"].sum() / credits_r

            print("【重修理想化分析】若所有<=79分课程都重修得90分:")
            print("GPA: %.4f | 算术平均: %.2f | 加权平均: %.2f" % (gpa_r, arith_r, weight_r))
            print("重修课程如下:")
            for _, row in resit_courses.iterrows():
                print(f"- {row['course_name']}")

            total_credits = self.total_graduation_credits
            past_points = df_resit[df_resit["is_predicted"] == False]["weighted_gpa"].sum()
            past_credits = df_resit[df_resit["is_predicted"] == False]["credit"].sum()
            pred_points = df_resit[df_resit["is_predicted"] == True]["weighted_gpa"].sum()
            pred_credits = df_resit[df_resit["is_predicted"] == True]["credit"].sum()
            unknown_credits = total_credits - past_credits - pred_credits
            target_total_points = self.target_gpa * total_credits
            remaining_points = target_total_points - past_points - pred_points

            if unknown_credits > 0:
                needed_gpa_unknown = remaining_points / unknown_credits
                print("【重修启用后，尚未规划学分】")
                print(
                    f"剩余学分: {unknown_credits:.1f} | 目标 GPA: {self.target_gpa} | 所需平均绩点: {needed_gpa_unknown:.4f}"
                )
                if needed_gpa_unknown > 5.0:
                    print("[严重警告] 即使重修所有低分，也无法通过常规方式达成目标！")
                elif needed_gpa_unknown > 4.5:
                    print("[难度极高] 未来几乎所有课都要冲击 95+。")
                elif needed_gpa_unknown > 4.0:
                    print("[难度高] 未来几乎所有课都要冲击 90+。")
                elif needed_gpa_unknown > 3.5:
                    print("[难度中等] 保持目前的预测水平(90+)即可稳步达成。")
                elif needed_gpa_unknown > 3.0:
                    print("[难度较低] 保持目前的预测水平(85+)即可稳步达成。")
                else:
                    print("[乐观] 达成目标有望，继续努力！")
            else:
                print("【重修启用后，尚未规划学分】剩余学分: 0 | 说明: 已与毕业总学分对齐。")
        else:
            print("【重修分析】没有分数低于79的科目。")
        print("-" * 30)

        target_total_points = self.target_gpa * self.total_graduation_credits
        needed_points_from_remaining = target_total_points - total_recorded_points

        if remaining_unknown_credits > 0:
            needed_avg_gpa = needed_points_from_remaining / remaining_unknown_credits
            print("【毕业目标，当前规划下剩余学分】")
            print(
                f"剩余学分: {remaining_unknown_credits:.1f} | 目标 GPA: {self.target_gpa} | 所需平均绩点: {needed_avg_gpa:.4f}"
            )

            if needed_avg_gpa > 5.0:
                print("[严重警告] 目标已无法通过常规方式达成，请考虑重修或调低预期。")
            elif needed_avg_gpa > 4.5:
                print("[难度极高] 未来几乎所有课都要冲击 95+。")
            elif needed_avg_gpa > 4.0:
                print("[难度高] 未来几乎所有课都要冲击 90+。")
            elif needed_avg_gpa > 3.5:
                print("[难度中等] 保持目前的预测水平(90+)即可稳步达成。")
            elif needed_avg_gpa > 3.0:
                print("[难度较低] 保持目前的预测水平(85+)即可稳步达成。")

        else:
            print("【毕业规划】已规划学分已达毕业要求总学分，无剩余未规划学分。")

        return df

    def visualize_bar(self, df):
        """按学分权重降序显示成绩柱状图。"""
        plt.figure(figsize=(15, 7))
        df_sorted = df.sort_values(by="credit", ascending=False)
        colors = ["#2E7D32" if not x else "#FB8C00" for x in df_sorted["is_predicted"]]

        bars = plt.bar(df_sorted["course_name"], df_sorted["score"], color=colors)
        plt.ylim(60, 100)
        plt.axhline(y=90, color="red", linestyle="--", alpha=0.3)

        plt.title("课程成绩分布图 (按学分权重从高到低排列)", fontsize=14)
        plt.ylabel("百分制分数")
        plt.xticks(rotation=45, ha="right", fontsize=9)

        for i, bar in enumerate(bars):
            gpa = self._score_to_gpa(bar.get_height())
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f"{gpa}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        plt.legend(["90分线", "已修读课程", "预测/在读课程"])
        plt.tight_layout()
        plt.show()


def main():
    print()
    print("======== ZJNU GPA 计算器 ========")
    print()

    target_gpa = prompt_float("目标毕业平均绩点", default=4.1)
    total_grad = prompt_float("毕业要求总学分", default=174.5)

    calculator = ZJNUGPACalculator(
        target_gpa=target_gpa, total_graduation_credits=total_grad
    )
    for n, c, s, p in COURSES_DATA:
        calculator.add_course(n, c, s, p)

    df_result = calculator.generate_report()
    draw = input("显示柱状图？(y/n，默认 y): ").strip().lower()
    if draw in ("", "y", "yes", "是"):
        calculator.visualize_bar(df_result)


if __name__ == "__main__":
    main()
