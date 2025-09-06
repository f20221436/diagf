import threading
import time
import csv
import psutil
import os
import matplotlib.pyplot as plt
from datetime import datetime


# GPU monitoring is disabled (CPU-only mode)
# try:
#     import pynvml
#     pynvml.nvmlInit()
#     NVML_AVAILABLE = True
# except ImportError:
#     NVML_AVAILABLE = False

class ResourceMonitor:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._log_path = None
        self._interval = 1

    # GPU monitoring is disabled (CPU-only mode)
    def _get_gpu_stats(self):
        return None, None

    def _log_worker(self):
        process = psutil.Process(os.getpid())
        with open(self._log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cpu_percent', 'ram_used_mb'])
            while not self._stop_event.is_set():
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cpu = process.cpu_percent(interval=None)
                ram = process.memory_info().rss / (1024 * 1024)
                writer.writerow([
                    ts,
                    cpu,
                    ram
                ])
                f.flush()
                time.sleep(self._interval)

    def start_logging(self, log_path, interval=1):
        self._log_path = log_path
        self._interval = interval
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._log_worker, daemon=True)
        self._thread.start()

    def stop_logging(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    @staticmethod
    def plot_metrics(log_path, save_path=None):
        import pandas as pd
        df = pd.read_csv(log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(df['timestamp'], df['cpu_percent'], label='CPU %')
        plt.ylabel('CPU %')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(df['timestamp'], df['ram_used_mb'], label='RAM Used (MB)', color='orange')
        plt.ylabel('RAM (MB)')
        plt.legend()
        plt.xlabel('Time')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
