import pandas as pd
import matplotlib.pyplot as plt

def plot_resource_usage(log_path='resource_log.csv'):
    df = pd.read_csv(log_path)
    if 'time' in df.columns:
        df['rel_time'] = df['time'] - df['time'].iloc[0]
        x = df['rel_time']
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Normalize to seconds from 0
        df['rel_time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        x = df['rel_time']
    else:
        x = range(len(df))

    # Compute and print summary statistics
    def print_stats(label, series, x_axis):
        avg = series.mean()
        minv = series.min()
        maxv = series.max()
        peak_idx = series.idxmax()
        peak_time = x_axis[peak_idx] if hasattr(x_axis, '__getitem__') else peak_idx
        print(f"{label}:")
        print(f"  Average: {avg:.2f}")
        print(f"  Min: {minv:.2f}")
        print(f"  Max: {maxv:.2f} (Peak at {peak_time})")

    print("\nResource Usage Summary:")
    if 'cpu_total' in df.columns:
        print_stats('CPU Total (%)', df['cpu_total'], x)
    elif 'cpu_percent' in df.columns:
        print_stats('CPU (%)', df['cpu_percent'], x)
    if 'mem_used' in df.columns:
        print_stats('Memory Used (MB)', df['mem_used'], x)
    elif 'ram_used_mb' in df.columns:
        print_stats('RAM Used (MB)', df['ram_used_mb'], x)

    plt.figure(figsize=(10, 6))
    if 'cpu_total' in df.columns:
        plt.subplot(2,1,1)
        plt.plot(x, df['cpu_total'], label='CPU Total (%)')
        plt.ylabel('CPU %')
        plt.legend()
    elif 'cpu_percent' in df.columns:
        plt.subplot(2,1,1)
        plt.plot(x, df['cpu_percent'], label='CPU %')
        plt.ylabel('CPU %')
        plt.legend()
    if 'mem_used' in df.columns:
        plt.subplot(2,1,2)
        plt.plot(x, df['mem_used'], label='Memory Used (MB)')
        if 'mem_available' in df.columns:
            plt.plot(x, df['mem_available'], label='Memory Available (MB)')
        plt.ylabel('Memory (MB)')
        plt.legend()
    elif 'ram_used_mb' in df.columns:
        plt.subplot(2,1,2)
        plt.plot(x, df['ram_used_mb'], label='RAM Used (MB)', color='orange')
        plt.ylabel('RAM (MB)')
        plt.legend()
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig('resource_usage_plot.png')
    plt.show()

if __name__ == '__main__':
    plot_resource_usage('resource_log.csv')
