def calculate_fcfs(processes):
    n = len(processes)
    waiting_time = [0] * n
    for i in range(1, n):
        waiting_time[i] = processes[i - 1]["burst"] + waiting_time[i - 1]
    return waiting_time


def calculate_sjf(processes):
    n = len(processes)
    proc_sorted = sorted(processes, key=lambda p: p["burst"])
    wt_sorted = [0] * n
    for i in range(1, n):
        wt_sorted[i] = proc_sorted[i - 1]["burst"] + wt_sorted[i - 1]
    waiting_time = [0] * n
    for i in range(n):
        original_index = proc_sorted[i]["id"] - 1
        waiting_time[original_index] = wt_sorted[i]
    return waiting_time


def calculate_priority_scheduling(processes):
    n = len(processes)
    proc_sorted = sorted(processes, key=lambda p: p["priority"])
    wt_sorted = [0] * n
    for i in range(1, n):
        wt_sorted[i] = proc_sorted[i - 1]["burst"] + wt_sorted[i - 1]
    waiting_time = [0] * n
    for i in range(n):
        original_index = proc_sorted[i]["id"] - 1
        waiting_time[original_index] = wt_sorted[i]
    return waiting_time


def calculate_round_robin(processes, quantum):
    n = len(processes)
    rem_burst_time = [p["burst"] for p in processes]
    waiting_time = [0] * n
    time = 0
    while True:
        done = True
        for i in range(n):
            if rem_burst_time[i] > 0:
                done = False
                if rem_burst_time[i] > quantum:
                    time += quantum
                    rem_burst_time[i] -= quantum
                else:
                    time += rem_burst_time[i]
                    waiting_time[i] = time - processes[i]["burst"]
                    rem_burst_time[i] = 0
        if done:
            break
    return waiting_time


def print_results(algorithm_name, processes, waiting_time, nim, nama):
    n = len(processes)
    total_waiting_time = sum(waiting_time)
    awt = total_waiting_time / n
    print("========================================")
    print(f"Algoritma {algorithm_name}")
    print(f"NIM: {nim}, Nama: {nama}")
    print("----------------------------------------")
    print(" Proses\tWaktu Tunggu")
    for i in range(n):
        print(f" P{processes[i]['id']}\t{waiting_time[i]}")
    print(f"\nTotal Waktu Tunggu = {total_waiting_time}")
    print(f"Average Waiting Time = {total_waiting_time}/{n} = {awt:.2f}")
    print("========================================\n")


if __name__ == "__main__":
    NIM = "123230099"
    NAMA = "AyodyaEnhanayoan"
    process_data = [
        {"id": 1, "burst": 17, "priority": 1},
        {"id": 2, "burst": 11, "priority": 3},
        {"id": 3, "burst": 7, "priority": 2},
        {"id": 4, "burst": 5, "priority": 4},
        {"id": 5, "burst": 9, "priority": 5},
        {"id": 6, "burst": 4, "priority": 6},
    ]
    quantum_time = 5

    wt_fcfs = calculate_fcfs(process_data)
    print_results("FCFS (First Come First Serve)", process_data, wt_fcfs, NIM, NAMA)

    wt_sjf = calculate_sjf(process_data)
    print_results("SJF (Shortest Job First)", process_data, wt_sjf, NIM, NAMA)

    wt_ps = calculate_priority_scheduling(process_data)
    print_results("Priority Scheduling", process_data, wt_ps, NIM, NAMA)

    wt_rr = calculate_round_robin(process_data, quantum_time)
    print_results(
        f"Round Robin (Quantum={quantum_time})", process_data, wt_rr, NIM, NAMA
    )
