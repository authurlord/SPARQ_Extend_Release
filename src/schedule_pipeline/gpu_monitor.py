import threading, time
import pynvml as nv

class GPUMonitor:
    def __init__(self, interval_sec=1.0):
        self.interval = interval_sec
        self._stop = False
        self._t = None
        self.samples = []  # (ts, gpu_index, util_pct, mem_used_MiB)

    def _loop(self):
        nv.nvmlInit()
        try:
            while not self._stop:
                n = nv.nvmlDeviceGetCount()
                ts = time.time()
                for i in range(n):
                    h = nv.nvmlDeviceGetHandleByIndex(i)
                    util = nv.nvmlDeviceGetUtilizationRates(h).gpu  # %
                    mem = nv.nvmlDeviceGetMemoryInfo(h).used // (1024*1024)  # MiB
                    self.samples.append((ts, i, util, mem))
                time.sleep(self.interval)
        finally:
            nv.nvmlShutdown()

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop = True
        if self._t:
            self._t.join()

    def avg_util(self):
        if not self.samples:
            return {}, 0.0
        per = {}
        for _, idx, util, _ in self.samples:
            per.setdefault(idx, []).append(util)
        per_avg = {idx: sum(v)/len(v) for idx, v in per.items()}
        # only count the gpu which utilization is not 0.0
        per_avg = {idx: v for idx, v in per_avg.items() if v > 0.0}
        overall = (sum(per_avg.values()) / len(per_avg)) if per_avg else 0.0
        return per_avg, overall



if __name__ == '__main__':
  # 用法：在你的整段执行前后包一层
  mon = GPUMonitor(interval_sec=1.0)
  mon.start()
  try:
    pass
  finally:
      mon.stop()
      per_avg, overall = mon.avg_util()
      print("Per-GPU avg util (%):", per_avg)
      print("Overall avg util (%):", overall)