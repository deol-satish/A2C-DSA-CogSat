# run_all_configs.py
import subprocess
import os
import time

channel_size_list = [10, 15, 20, 25, 30]
leo_users_list = [10, 15, 20]
geo_users_list = [10, 15, 20]

MAX_PARALLEL_JOBS = 4  # Define the maximum number of parallel jobs

count = 0
running_processes = [] # List to keep track of currently running Popen objects

print("Starting all A2C training configurations with parallel execution...")
print(f"Maximum concurrent jobs: {MAX_PARALLEL_JOBS}")
print("-" * 50)

# Generate all configurations first to easily batch them
all_configs = []
for channel_size in channel_size_list:
    for leo_users in leo_users_list:
        for geo_users in geo_users_list:
            all_configs.append({
                "channel_size": channel_size,
                "leo_users": leo_users,
                "geo_users": geo_users
            })

total_configs = len(all_configs)
config_index = 0

while config_index < total_configs or running_processes:
    # Start new jobs until MAX_PARALLEL_JOBS are running or all configs are processed
    while len(running_processes) < MAX_PARALLEL_JOBS and config_index < total_configs:
        current_config = all_configs[config_index]
        channel_size = current_config["channel_size"]
        leo_users = current_config["leo_users"]
        geo_users = current_config["geo_users"]

        cmd = [
            "python", "train_a2c.py",
            "--channels", str(channel_size),
            "--leo_users", str(leo_users),
            "--geo_users", str(geo_users),
            "--seed", str(42)
        ]
        
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file_name = f"a2c_{channel_size}_{leo_users}_{geo_users}.log"
        log_file_path = os.path.join(log_dir, log_file_name)

        print(f"[{count + 1}/{total_configs}] Starting training for config:")
        print(f"  Channels: {channel_size}, LEO Users: {leo_users}, GEO Users: {geo_users}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Logging to: {log_file_path}")
        
        try:
            with open(log_file_path, "w") as f:
                process = subprocess.Popen(cmd, stdout=f, stderr=f, text=True)
                running_processes.append((process, log_file_path, current_config)) # Store process object, log path, and config for feedback
            print(f"  Successfully initiated process for config: {channel_size}-{leo_users}-{geo_users}")
        except Exception as e:
            print(f"  ERROR: Failed to start process for config {channel_size}-{leo_users}-{geo_users}: {e}")
        
        print("-" * 50)
        count += 1
        config_index += 1

    # If all configs are launched, wait for remaining processes to finish
    if config_index == total_configs and not running_processes:
        break # All done

    # Wait for at least one process to complete if we have MAX_PARALLEL_JOBS running
    # or if we are waiting for the last few jobs.
    if running_processes:
        print(f"Waiting for {len(running_processes)} active jobs to complete before launching more...")
        # Use poll to check status without blocking indefinitely, or wait for all in the batch
        
        # We'll wait for ALL currently running processes to finish before moving to the next batch.
        # This is the "wait for it to complete before going to next 4" logic.
        completed_batch = []
        for p, log_path, config in running_processes:
            p.wait() # This will block until the process finishes
            print(f"  Job completed for config: Channels={config['channel_size']}, LEO={config['leo_users']}, GEO={config['geo_users']} (Log: {log_path})")
            completed_batch.append((p, log_path, config))
        
        # Clear the list of running processes after the batch completes
        running_processes = []
        print(f"Batch of {len(completed_batch)} jobs completed. Moving to next batch (if any).")
        print("=" * 70) # Separator for batches
    
    # Small delay to prevent busy-waiting if processes are very fast
    time.sleep(0.5)

print(f"\nTotal jobs processed: {count}")
print("All A2C training configurations have been executed.")
print("Check 'logs/' directory for output of each run.")