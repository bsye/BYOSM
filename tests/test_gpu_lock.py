import threading
import time

from byosm.train.gpu import gpu_slot


def test_only_one_job_trains_on_the_gpu_at_a_time(ws):
    order = []

    def job(name):
        with gpu_slot("mps", log=lambda _: None):
            order.append(f"{name} in")
            time.sleep(0.3)
            order.append(f"{name} out")

    import byosm.train.gpu as gpu

    gpu.WAIT_S = 0.05
    threads = [threading.Thread(target=job, args=(n,)) for n in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert order[0].endswith("in") and order[1] == order[0].replace("in", "out")


def test_the_cpu_needs_no_slot(ws):
    with gpu_slot("cpu", log=lambda _: None), gpu_slot("cpu", log=lambda _: None):
        pass
