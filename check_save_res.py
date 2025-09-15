import importlib.util, inspect, os, sys
path = r"C:\Users\DEVESH PALO\projects\Diagf\diagf\transforms\events\metric_event.py"
print("Checking path:", path)
spec = importlib.util.spec_from_file_location("local_metric_event", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("Loaded file:", getattr(m, '__file__', None))
print("MetricEvent defined?:", hasattr(m, 'MetricEvent'))
if hasattr(m, 'MetricEvent'):
    print("save_res on class?:", hasattr(m.MetricEvent, 'save_res'))
    if hasattr(m.MetricEvent, 'save_res'):
        print("--- save_res source (first 20 lines) ---")
        src = inspect.getsource(m.MetricEvent.save_res).splitlines()
        for i, line in enumerate(src[:20], 1):
            print(f"{i:03d}: {line}")
