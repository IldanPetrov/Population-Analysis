# time_utils.py
def cycle_to_time(cycle: int) -> float:
    # dt_map = {
    #     (1, 101): 0.25,
    #     (101, 102): 1.5,
    #     (102, 103): 0.5,
    #     (103, float('inf')): 1
    # }
    dt_map = {
        (1, 101): 0.25,
        (101, float('inf')): 1
    }

    time = 0
    for (start, end), dt in dt_map.items():
        if cycle > end:
            time += (end - start) * dt
        else:
            time += (cycle - start) * dt
            break

    return time