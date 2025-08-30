import json
import pulp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# ---------------------------
# Đọc dữ liệu từ file
# ---------------------------
def read_json_files():
    with open("../../Loc_data_price/price.json", "r") as f:
        price_data = json.load(f)
    with open("../../Loc_solar/gop_solar.json", "r") as f:
        solar_data = json.load(f)
    with open("../final_gop.json", "r") as f:
        final_gop_data = json.load(f)
    with open("../max.json", "r") as f:
        max_data = json.load(f)
    return price_data, solar_data, final_gop_data, max_data

# ---------------------------
# Chuyển chuỗi thời gian sang datetime
# ---------------------------
def parse_time(time_str):
    # Ví dụ: "Thu, 26 Apr 2018 00:02:16 GMT"
    return datetime.strptime(time_str.replace("GMT", "").strip(), "%a, %d %b %Y %H:%M:%S")

# ---------------------------
# Lấy các phiên EV trong khoảng [day_start, day_end)
# Loại phiên nếu disconnectTime >= day_end và loại những phiên mà connectionTime không nằm trong khoảng
# ---------------------------
def get_sessions_for_period(final_gop_data, day_start, day_end):
    """
    Trả về các phiên sạc có connectionTime nằm trong [day_start, day_end)
    và disconnectTime nhỏ hơn day_end.
    """
    sessions_in_range = []
    for k, sessions in final_gop_data.items():
        for session in sessions:
            conn = parse_time(session["connectionTime"])
            disc = parse_time(session["disconnectTime"])
            # Yêu cầu connectionTime phải nằm trong khoảng [day_start, day_end)
            # và loại bỏ phiên nào có disconnectTime >= day_end
            if day_start <= conn < day_end and disc < day_end:
                sessions_in_range.append(session)
    return sessions_in_range

# ---------------------------
# Xây dựng dữ liệu cho 48 giờ (2 ngày)
# ---------------------------
def build_data_for_multiple_days(day_start, num_days, sessions, price_data, solar_data, max_data):
    """
    day_start: datetime (ví dụ 2018-04-26 00:00:00)
    num_days: số ngày (ví dụ 2)
    => Ta sẽ xét [day_start, day_end) = 48 tiếng
    """
    day_end = day_start + timedelta(days=num_days)
    T = num_days * 24  # 48 giờ (cho 2 ngày)

    # 1) Mảng giá điện p_grid
    p_grid = []
    current = day_start
    while current < day_end:
        day_str = current.strftime("%Y-%m-%d")
        hour_idx = current.hour
        price_for_day = price_data.get(day_str, [0]*24)  # phòng trường hợp không có key
        price_val = price_for_day[hour_idx] if hour_idx < len(price_for_day) else 0
        p_grid.append(price_val)
        current += timedelta(hours=1)

    # 2) Mảng năng lượng tái tạo R
    R_list = []
    current = day_start
    while current < day_end:
        solar_key = current.strftime("%Y%m%d")
        hour_idx = current.hour
        if solar_key in solar_data and hour_idx < len(solar_data[solar_key]):
            R_val = solar_data[solar_key][hour_idx]["R(i)"]
        else:
            R_val = 0
        R_list.append(R_val)
        current += timedelta(hours=1)

    # 3) Sắp xếp session theo connectionTime (FCFS)
    sessions_sorted = sorted(sessions, key=lambda s: parse_time(s["connectionTime"]))

    # 4) Tạo ma trận A[i][t], L_req
    A_matrix = []
    L_req = []
    conn_times = []

    for session in sessions_sorted:
        conn = parse_time(session["connectionTime"])
        disc = parse_time(session["disconnectTime"])

        # Giới hạn vào [day_start, day_end) để tính fraction
        session_start = max(conn, day_start)
        session_end = min(disc, day_end)

        availability = []
        for t in range(T):
            slot_start = day_start + timedelta(hours=t)
            slot_end = slot_start + timedelta(hours=1)
            eff_start = max(slot_start, session_start)
            eff_end = min(slot_end, session_end)
            if eff_end > eff_start:
                fraction = (eff_end - eff_start).total_seconds() / 3600.0
            else:
                fraction = 0
            availability.append(fraction)

        A_matrix.append(availability)
        L_req.append(session["kWhDelivered"])
        conn_times.append(conn)

    # 5) Lấy công suất, giới hạn lưới
    s = max_data["doubled_max_rate"]  # ví dụ 22 kW, 44 kW, ...
    C_grid = 300  # Tùy chỉnh
    data = {
        "T": T,
        "N": len(sessions_sorted),
        "A": A_matrix,
        "L_req": L_req,
        "conn_times": conn_times,
        "p_grid": p_grid,
        "R": R_list,
        "s": s,
        "eta": 0.9,
        "C_grid": C_grid,
        "delta_t": 1,
        "sessions_sorted": sessions_sorted
    }
    return data

# ---------------------------
# Xây dựng & giải LP (có dùng R)
# ---------------------------
def build_lp_model(data):
    T = data["T"]
    N = data["N"]
    A = data["A"]
    L_req = data["L_req"]
    s = data["s"]
    eta = data["eta"]
    p_grid = data["p_grid"]
    R_list = data["R"]
    C_grid = data["C_grid"]
    delta_t = data["delta_t"]

    problem = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMinimize)

    Y = pulp.LpVariable.dicts("Y", ((i, t) for i in range(N) for t in range(T)),
                              lowBound=0, cat=pulp.LpContinuous)
    S_plus = pulp.LpVariable.dicts("S_plus", (t for t in range(T)),
                                   lowBound=0, cat=pulp.LpContinuous)
    R_used = pulp.LpVariable.dicts("R_used", (t for t in range(T)),
                                   lowBound=0, cat=pulp.LpContinuous)

    # Hàm mục tiêu: chi phí mua điện lưới
    problem += pulp.lpSum([p_grid[t] * S_plus[t] * delta_t for t in range(T)]), "Minimize_Cost"

    # Ràng buộc năng lượng EV
    for i in range(N):
        problem += eta * pulp.lpSum([Y[(i, t)] for t in range(T)]) >= L_req[i], f"EnergyReq_EV_{i}"

    # Ràng buộc công suất (EV chỉ sạc khi có mặt, tối đa s kW)
    for i in range(N):
        for t in range(T):
            problem += Y[(i, t)] <= s * A[i][t], f"Presence_EV_{i}_t_{t}"

    # Ràng buộc lưới + tái tạo
    for t in range(T):
        total_load = pulp.lpSum([Y[(i, t)] for i in range(N)])
        problem += total_load - R_used[t] <= C_grid, f"GridLimit_t_{t}"
        problem += S_plus[t] >= total_load - R_used[t], f"SplusPositivity_t_{t}"
        problem += R_used[t] <= R_list[t], f"RenewableLimit_t_{t}"

    return problem, Y, S_plus, R_used

def solve_lp_model(problem):
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[problem.status]
    obj_val = pulp.value(problem.objective)
    return status, obj_val

# ---------------------------
# FCFS (không dùng R)
# ---------------------------
def do_fcfs_no_renewable(data):
    T = data["T"]
    N = data["N"]
    A = data["A"]
    L_req = data["L_req"]
    s = data["s"]
    eta = data["eta"]
    C_grid = data["C_grid"]
    conn_times = data["conn_times"]

    needed = [(L_req[i] / eta) for i in range(N)]  # lượng cần nạp trên lưới (chưa trừ R)
    X = [[0.0]*T for _ in range(N)]
    STATION_LIMIT = 80  # giới hạn số trạm

    grid_usage = [0.0]*T
    for t in range(T):
        grid_leftover = C_grid
        station_count = 0
        # Tìm session có nhu cầu và hiện diện
        present_sessions = []
        for i in range(N):
            if needed[i] > 1e-9 and A[i][t] > 0:
                present_sessions.append(i)
        # FCFS theo connectionTime
        present_sessions.sort(key=lambda i_: conn_times[i_])

        for i in present_sessions:
            if station_count < STATION_LIMIT and grid_leftover > 1e-9:
                deliverable = min(s*A[i][t], needed[i], grid_leftover)
                X[i][t] = deliverable
                needed[i] -= deliverable
                grid_leftover -= deliverable
                station_count += 1

        grid_usage[t] = sum(X[i][t] for i in range(N))

    return grid_usage

# ---------------------------
# MAIN
# ---------------------------
def main():
    # 1) Xác định ngày bắt đầu + số ngày (2 ngày => 48 giờ)
    target_day_str = "2018-09-06"
    end_date_str = "2018-09-07"
    num_days = 2
    day_start = datetime.strptime(target_day_str, "%Y-%m-%d")

    # 2) Đọc dữ liệu
    price_data, solar_data, final_gop_data, max_data = read_json_files()

    # 3) Lấy session trong [day_start, day_end) với yêu cầu connectionTime nằm trong khoảng
    day_end = day_start + timedelta(days=num_days)
    sessions_in_range = get_sessions_for_period(final_gop_data, day_start, day_end)

    if len(sessions_in_range) == 0:
        print("Không có phiên EV nào thoả mãn trong 2 ngày được chọn!")
        return

    # 4) Xây dựng data 48h
    data_2days = build_data_for_multiple_days(day_start, num_days, sessions_in_range,
                                              price_data, solar_data, max_data)

    # 5) Tính FCFS (không R)
    fcfs_usage = do_fcfs_no_renewable(data_2days)
    # Ghi ra file
    fcfs_result = []
    for t in range(data_2days["T"]):
        fcfs_result.append({
            "hour": t,
            "grid_usage": fcfs_usage[t]
        })
    with open("fcfs_2days.json", "w", encoding="utf-8") as f:
        json.dump(fcfs_result, f, ensure_ascii=False, indent=2)
    print("Đã ghi kết quả FCFS vào fcfs_2days.json")

    # 6) LP (có R)
    problem, Y, S_plus, R_used = build_lp_model(data_2days)
    status_lp, obj_val_lp = solve_lp_model(problem)
    print(f"LP Status: {status_lp}, Objective={obj_val_lp}")

    lp_usage = [pulp.value(S_plus[t]) for t in range(data_2days["T"])]
    # Ghi ra file
    lp_result = []
    for t in range(data_2days["T"]):
        lp_result.append({
            "hour": t,
            "grid_usage": lp_usage[t]
        })
    with open("lp_2days.json", "w", encoding="utf-8") as f:
        json.dump(lp_result, f, ensure_ascii=False, indent=2)
    print("Đã ghi kết quả LP vào lp_2days.json")

    # 7) Vẽ biểu đồ với 2 trục y:
    import matplotlib.ticker as ticker
    hours = range(data_2days["T"])

    fcfs_vals = fcfs_usage
    lp_vals = lp_usage
    price_vals = data_2days["p_grid"]  # Giá điện 48 giờ

    fig, ax1 = plt.subplots(figsize=(10,5))

    # Vẽ FCFS & LP trên trục y bên trái
    line_fcfs = ax1.plot(hours, fcfs_vals, label="FCFS without Solar Energy", marker='o', color='blue')
    line_lp = ax1.plot(hours, lp_vals, label="Our Model with Solar Energy", marker='s', color='orange')

    ax1.set_title(f"Total power allocated to EVs of 2 days: ({target_day_str}) and ({end_date_str}) in Caltech")
    ax1.set_xlabel("Hours (0 - 47)")
    ax1.set_ylabel("kWh from Grid")
    ax1.set_xticks(range(0, data_2days["T"]))
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.2f}"))
    ax1.grid(True, alpha=0.3)

    # Trục y thứ 2 (bên phải) để hiển thị giá điện
    ax2 = ax1.twinx()
    line_price = ax2.plot(hours, price_vals, label="Electricity price", linestyle='--', color='red')
    ax2.set_ylabel("Electricity price (Euro/MWh)")

    # Gom tất cả lines để legend chung
    lines = line_fcfs + line_lp + line_price
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
