import json
import pulp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import ticker
import os

# ---------------------------
# Hàm đọc dữ liệu từ file
# ---------------------------
def read_json_files():
    with open("../Loc_data_price/price.json", "r") as f:
        price_data = json.load(f)
    with open("../Loc_solar/gop_solar.json", "r") as f:
        solar_data = json.load(f)
    with open("./final_gop.json", "r") as f:
        final_gop_data = json.load(f)
    with open("./max.json", "r") as f:
        max_data = json.load(f)
    return price_data, solar_data, final_gop_data, max_data

# ---------------------------
# Hàm chuyển đổi chuỗi thời gian sang datetime
# ---------------------------
def parse_time(time_str):
    # Định dạng: "Thu, 26 Apr 2018 00:02:16 GMT"
    return datetime.strptime(time_str.replace("GMT", "").strip(), "%a, %d %b %Y %H:%M:%S")

# ---------------------------
# Lấy các ngày từ final_gop_data nằm trong khoảng ngày cần xử lý
# ---------------------------
def parse_final_gop_dates(final_gop_data, start_date, end_date):
    mapping = {}
    for key in final_gop_data:
        try:
            dt = datetime.strptime(key, "%d-%b-%Y")
        except Exception:
            continue
        if start_date <= dt <= end_date:
            mapping[dt] = key
    sorted_dates = sorted(mapping.keys())
    return sorted_dates, mapping

# ---------------------------
# Tạo các block overlapping
# ---------------------------
def create_overlapping_blocks(sorted_dates, final_gop_data, mapping):
    blocks = []
    pending = []
    if len(sorted_dates) < 2:
        # Nếu chỉ có 1 ngày, tạo block 24h
        block = {"dates": [sorted_dates[0]], "sessions": final_gop_data[mapping[sorted_dates[0]]]}
        blocks.append(block)
        return blocks

    for i in range(1, len(sorted_dates)):
        day_prev = sorted_dates[i-1]
        day_curr = sorted_dates[i]
        block_dates = [day_prev, day_curr]
        block_start = datetime.combine(day_prev, datetime.min.time())
        block_end = datetime.combine(day_curr, datetime.min.time()) + timedelta(days=1)

        sessions_prev = final_gop_data.get(mapping[day_prev], [])
        sessions_curr = final_gop_data.get(mapping[day_curr], [])
        block_sessions = pending + sessions_prev + sessions_curr

        new_block_sessions = []
        new_pending = []
        for session in block_sessions:
            dtime = parse_time(session["disconnectTime"])
            if dtime >= block_end:
                new_pending.append(session)
            else:
                new_block_sessions.append(session)

        block = {"dates": block_dates, "sessions": new_block_sessions}
        blocks.append(block)
        pending = new_pending

    if pending:
        last_date = sorted_dates[-1]
        block_start = datetime.combine(last_date, datetime.min.time())
        block_end = block_start + timedelta(days=1)
        final_sessions = []
        new_pending = []
        for session in pending:
            dtime = parse_time(session["disconnectTime"])
            if dtime >= block_end:
                new_pending.append(session)
            else:
                final_sessions.append(session)
        block = {"dates": [last_date], "sessions": final_sessions}
        blocks.append(block)
        pending = new_pending
    return blocks

# ---------------------------
# Xây dựng dữ liệu đầu vào (bao gồm ma trận A) cho mỗi block
# ---------------------------
def get_block_data_from_block(block, price_data, solar_data, max_data):
    block_dates = block["dates"]
    sessions = block["sessions"]
    block_start = datetime.combine(block_dates[0], datetime.min.time())
    if len(block_dates) > 1:
        block_end = datetime.combine(block_dates[-1], datetime.min.time()) + timedelta(days=1)
    else:
        block_end = block_start + timedelta(days=1)
    T = int((block_end - block_start).total_seconds() / 3600)

    # p_grid
    p_grid = []
    current = block_start
    while current < block_end:
        day_str = current.strftime("%Y-%m-%d")
        hour_index = current.hour
        price = price_data.get(day_str, [0]*24)[hour_index] if day_str in price_data else 0
        price = price / 1000
        p_grid.append(price)
        current += timedelta(hours=1)

    # R
    R = []
    current = block_start
    while current < block_end:
        solar_key = current.strftime("%Y%m%d")
        hour_index = current.hour
        if solar_key in solar_data:
            R_val = solar_data[solar_key][hour_index]["R(i)"]
        else:
            R_val = 0
        R.append(R_val)
        current += timedelta(hours=1)

    # Sort session theo connectionTime (FCFS)
    sessions_sorted = sorted(sessions, key=lambda s: parse_time(s["connectionTime"]))

    A_matrix = []
    L_req = []
    conn_times = []

    for session in sessions_sorted:
        conn = parse_time(session["connectionTime"])
        disc = parse_time(session["disconnectTime"])
        session_start = max(conn, block_start)
        session_end = min(disc, block_end)
        availability = []
        for t in range(T):
            slot_start = block_start + timedelta(hours=t)
            slot_end = slot_start + timedelta(hours=1)
            eff_start = max(slot_start, session_start)
            eff_end = min(slot_end, session_end)
            if eff_end > eff_start:
                fraction = (eff_end - eff_start).total_seconds() / 3600.0
                fraction = min(fraction, 1)
            else:
                fraction = 0
            availability.append(fraction)
        A_matrix.append(availability)
        L_req.append(session["kWhDelivered"])
        conn_times.append(conn)

    s = max_data["doubled_max_rate"]
    data = {
        "T": T,
        "N": len(sessions_sorted),
        "sessions_sorted": sessions_sorted,
        "A": A_matrix,
        "L_req": L_req,
        "conn_times": conn_times,
        "p_grid": p_grid,
        "R": R,
        "s": s,
        "eta": 0.9,
        "C_grid": 300,  # Rất lớn, tùy chỉnh nếu cần
        "delta_t": 1
    }
    return data

# ---------------------------
# 1) HÀM GIẢI TỐI ƯU (solver) - giữ nguyên hoặc tùy bạn
# ---------------------------
def build_model(data):
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

    # Hàm mục tiêu
    problem += pulp.lpSum([p_grid[t] * S_plus[t] * delta_t for t in range(T)]), "Minimize_Cost"

    # Ràng buộc năng lượng tối thiểu
    for i in range(N):
        T_i = [t for t in range(T) if A[i][t] > 0]
        problem += eta * pulp.lpSum([Y[(i, t)] * delta_t for t in T_i]) >= L_req[i], f"EnergyReq_EV_{i}"

    # Ràng buộc công suất
    for i in range(N):
        for t in range(T):
            problem += Y[(i, t)] <= s, f"MaxPower_EV_{i}_t_{t}"
            problem += Y[(i, t)] <= s * A[i][t], f"Presence_EV_{i}_t_{t}"

    # Ràng buộc lưới + tái tạo
    for t in range(T):
        total_load = pulp.lpSum([Y[(i, t)] for i in range(N)])
        problem += total_load - R_used[t] <= C_grid, f"GridLimit_t_{t}"
        problem += S_plus[t] >= total_load - R_used[t], f"SplusPositivity_t_{t}"
        problem += R_used[t] <= R_list[t], f"RenewableLimit_t_{t}"

    return problem, Y, S_plus, R_used

def solve_model(problem):
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[problem.status]
    obj_val = pulp.value(problem.objective)
    return status, obj_val

# ---------------------------
# 2) HÀM GIẢ LẬP FCFS CÓ RÀNG BUỘC C_GRID
# ---------------------------
def do_fcfs_block(data):
    """
    Thực hiện cơ chế sạc First-Come-First-Serve với tối đa 80 trạm sạc (mỗi trạm công suất s).
    Đồng thời giới hạn công suất lưới C_grid mỗi giờ.
    Áp dụng công thức: X_{i,t} = min( s*A[i][t], needed[i], leftover_grid )
    """
    T = data["T"]
    N = data["N"]
    A = data["A"]            # A[i][t] = phần trăm giờ EV i có mặt trong giờ t
    L_req = data["L_req"]    # kWh EV i cần nạp (đầu ra)
    s = data["s"]            # công suất tối đa (kW) của mỗi trạm
    eta = data["eta"]        # hiệu suất sạc
    p_grid = data["p_grid"]  # giá điện
    R_list = data["R"]       # năng lượng tái tạo
    C_grid = data["C_grid"]  # công suất lưới tối đa (kWh/h) mỗi giờ
    conn_times = data["conn_times"]

    needed = [(L_req[i] / eta) for i in range(N)]  # kWh trạm cần cung cấp để xe i nhận đủ L_req[i] sau hiệu suất
    X = [[0.0]*T for _ in range(N)]                # Ma trận kết quả sạc (kWh)
    STATION_LIMIT = 80                             # Tối đa 80 trạm mỗi giờ

    # Lặp qua từng giờ
    for t in range(T):
        # Công suất lưới còn lại giờ t (kWh)
        grid_leftover = C_grid
        station_count = 0

        # Tìm các phiên i còn cần năng lượng, đang hiện diện giờ t
        present_sessions = []
        for i in range(N):
            if needed[i] > 1e-9 and A[i][t] > 0:
                present_sessions.append(i)

        # Sắp xếp theo connectionTime (First Come)
        present_sessions.sort(key=lambda i_: conn_times[i_])

        for i in present_sessions:
            # Nếu còn trạm và còn lưới
            if station_count < STATION_LIMIT and grid_leftover > 1e-9:
                # Lượng tối đa phiên i có thể nhận trong giờ t
                # s*A[i][t]: công suất trạm * tỉ lệ thời gian
                # needed[i]: EV i còn cần
                # grid_leftover: lưới còn lại
                deliverable = min(s*A[i][t], needed[i], grid_leftover)

                X[i][t] = deliverable
                needed[i] -= deliverable
                grid_leftover -= deliverable
                station_count += 1
            else:
                # Hết trạm hoặc hết công suất lưới
                break

    # Tính chi phí
    total_cost = 0.0
    for t in range(T):
        total_load_t = sum(X[i][t] for i in range(N))  # Tổng kWh sạc trong giờ t
        used_R_t = min(R_list[t], total_load_t)        # Dùng tái tạo trước
        cost_t = p_grid[t] * (total_load_t - used_R_t) # Phần còn lại mua lưới
        total_cost += cost_t

    return total_cost

# ---------------------------
# Main (chạy thử so sánh)
# ---------------------------
def main():
    # 1) Khoảng ngày
    start_date = datetime.strptime("25-04-2018", "%d-%m-%Y")
    end_date = datetime.strptime("31-12-2019", "%d-%m-%Y")
    
    price_data, solar_data, final_gop_data, max_data = read_json_files()
    
    # 2) Lọc dữ liệu
    filtered_price = {}
    for k, v in price_data.items():
        try:
            dt = datetime.strptime(k, "%Y-%m-%d")
            if start_date <= dt <= end_date:
                filtered_price[k] = v
        except Exception:
            continue
    price_data = filtered_price

    filtered_solar = {}
    for k, v in solar_data.items():
        try:
            dt = datetime.strptime(k, "%Y%m%d")
            if start_date <= dt <= end_date:
                filtered_solar[k] = v
        except Exception:
            continue
    solar_data = filtered_solar

    # 3) Tạo block
    sorted_dates, mapping = parse_final_gop_dates(final_gop_data, start_date, end_date)
    if not sorted_dates:
        print("Không có dữ liệu trong khoảng ngày được chọn!")
        return

    blocks = create_overlapping_blocks(sorted_dates, final_gop_data, mapping)
    print(f"Tổng số block cần xử lý: {len(blocks)}")

    # 4) File ghi kết quả
    jsonl_file_opt = "results.jsonl"   # Kết quả tối ưu
    jsonl_file_fcfs = "FCFS.jsonl"     # Kết quả FCFS
    f_opt = open(jsonl_file_opt, "a", encoding="utf-8")
    f_fcfs = open(jsonl_file_fcfs, "a", encoding="utf-8")
    
    monthly_results_opt = {}
    monthly_results_fcfs = {}

    # 5) Xử lý block
    for idx, block in enumerate(blocks, 1):
        block_dates = block["dates"]
        if len(block_dates) == 2:
            date_range_str = f"{block_dates[0].strftime('%Y-%m-%d')} to {block_dates[1].strftime('%Y-%m-%d')}"
        else:
            date_range_str = f"{block_dates[0].strftime('%Y-%m-%d')}"
        print("\n================================================")
        print(f"Đang xử lý block {idx}: {date_range_str}")
        
        data = get_block_data_from_block(block, price_data, solar_data, max_data)
        T, N = data["T"], data["N"]
        print(f" - Số giờ trong block (T): {T}")
        print(f" - Số phiên EV (N): {N}")

        if data["p_grid"]:
            print(f" - Giá điện mẫu (p_grid): {data['p_grid'][:5]} ...")
        if data["R"]:
            print(f" - Năng lượng tái tạo mẫu (R): {data['R'][:5]} ...")
        print(f" - Năng lượng yêu cầu (L_req): {data['L_req']}")

        if N == 0:
            # Không có phiên
            print("Không có phiên sạc nào trong block này, bỏ qua!")
            result_obj_opt = {
                "date_range": date_range_str,
                "objective_value": None,
                "status": "No session"
            }
            f_opt.write(json.dumps(result_obj_opt) + "\n")
            f_opt.flush()

            result_obj_fcfs = {
                "date_range": date_range_str,
                "fcfs_cost": None,
                "status": "No session"
            }
            f_fcfs.write(json.dumps(result_obj_fcfs) + "\n")
            f_fcfs.flush()
            continue
        
        # 5.1) Giải tối ưu
        problem, Y, S_plus, R_used = build_model(data)
        status_opt, obj_val_opt = solve_model(problem)
        print(f" - Solver: {status_opt} với objective value = {obj_val_opt}")
        
        result_obj_opt = {
            "date_range": date_range_str,
            "objective_value": obj_val_opt,
            "status": status_opt
        }
        f_opt.write(json.dumps(result_obj_opt) + "\n")
        f_opt.flush()

        # 5.2) Giải FCFS
        fcfs_cost = do_fcfs_block(data)
        print(f" - FCFS cost = {fcfs_cost}")

        result_obj_fcfs = {
            "date_range": date_range_str,
            "fcfs_cost": fcfs_cost,
            "status": "Done"
        }
        f_fcfs.write(json.dumps(result_obj_fcfs) + "\n")
        f_fcfs.flush()

        # 5.3) Cộng dồn theo tháng
        month_str = block_dates[0].strftime("%Y-%m")
        if obj_val_opt is not None:
            monthly_results_opt[month_str] = monthly_results_opt.get(month_str, 0) + obj_val_opt
        if fcfs_cost is not None:
            monthly_results_fcfs[month_str] = monthly_results_fcfs.get(month_str, 0) + fcfs_cost

    f_opt.close()
    f_fcfs.close()

    # 6) In tổng hợp
    print("\n======== TỔNG HỢP THEO THÁNG (Tối Ưu) ========")
    for month, cost in sorted(monthly_results_opt.items()):
        print(f" - {month}: {cost}")

    print("\n======== TỔNG HỢP THEO THÁNG (FCFS) ========")
    for month, cost in sorted(monthly_results_fcfs.items()):
        print(f" - {month}: {cost}")

    # 7) Vẽ biểu đồ so sánh
    months_opt = sorted(monthly_results_opt.keys())
    costs_opt = [monthly_results_opt[m] for m in months_opt]

    plt.figure(figsize=(8,4))
    plt.bar(months_opt, costs_opt, color='green', alpha=0.7)
    plt.xlabel("Tháng")
    plt.ylabel("Chi Phí Điện Lưới (Optimized)")
    plt.title("Chi phí tối ưu theo tháng")
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    plt.show()

    months_fcfs = sorted(monthly_results_fcfs.keys())
    costs_fcfs = [monthly_results_fcfs[m] for m in months_fcfs]

    plt.figure(figsize=(8,4))
    plt.bar(months_fcfs, costs_fcfs, color='orange', alpha=0.7)
    plt.xlabel("Tháng")
    plt.ylabel("Chi Phí Điện Lưới (FCFS)")
    plt.title("Chi phí FCFS theo tháng")
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
