import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib import ticker

def read_jsonl(filename, key_name):
    """Đọc file JSONL và tổng hợp dữ liệu theo tháng."""
    monthly_data = defaultdict(float)
    
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                date_range = entry["date_range"]  # "YYYY-MM-DD to YYYY-MM-DD" hoặc "YYYY-MM-DD"
                value = entry.get(key_name, None)
                if value is None:
                    continue
                first_date = date_range.split(" to ")[0]
                month_str = first_date[:7]
                monthly_data[month_str] += value
            except json.JSONDecodeError:
                print("Lỗi đọc dòng:", line.strip())
    return monthly_data

# Đọc dữ liệu từ JSONL
optimized_data = read_jsonl("results.jsonl", "objective_value")
fcfs_data = read_jsonl("FCFS.jsonl", "fcfs_cost")

# Danh sách tất cả các tháng
all_months = sorted(set(optimized_data.keys()).union(set(fcfs_data.keys())))

# Dữ liệu theo tháng
optimized_costs = [optimized_data.get(month, 0) for month in all_months]
fcfs_costs = [fcfs_data.get(month, 0) for month in all_months]

# Thiết lập vị trí cho các cột
bar_width = 0.35  # Độ rộng của mỗi cột
index = np.arange(len(all_months))  # Tạo mảng các vị trí cho các tháng

# Vẽ biểu đồ cột
plt.figure(figsize=(10, 6))
plt.bar(index, optimized_costs, bar_width, color="green", label="Our Model")
plt.bar(index + bar_width, fcfs_costs, bar_width, color="orange", label="FCFS")

# Định dạng biểu đồ
plt.xlabel("Months")
plt.ylabel("Total Cost (EURO)")
plt.title("Electricity Costs: Our Model vs FCFS by Month in Caltech")
plt.xticks(index + bar_width / 2, all_months, rotation=45)  # Đặt nhãn tháng ở giữa hai cột
plt.legend()
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout()
plt.show()