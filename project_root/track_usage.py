import os
import subprocess
import psutil
import pynvml
import sys
import time

# Function to get CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

# Function to get GPU usage
def get_gpu_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming there's only one GPU
    info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return info.gpu

def get_RAM_usage():
    return psutil.virtual_memory().percent

# Function to run the target Python script with an integer argument
def run_target_script(script_path, argument):
    return subprocess.Popen(['python', script_path, str(argument)])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python track_usage.py <path_to_target_script.py> <integer_argument>")
        sys.exit(1)
   
    # Extract the path to the target script and the integer argument from the command line
    target_script_path = sys.argv[1]
    try:
        argument = int(sys.argv[2])
    except ValueError:
        print("Argument must be an integer")
        sys.exit(1)

    # Run the target script with the integer argument
    process = run_target_script(target_script_path, argument)
    start_time = time.time()

    # Initialize cumulative CPU and GPU usage
    total_cpu_usage = 0
    total_gpu_usage = 0
    total_ram_usage = 0
    count = 0

    # Monitoring loop
    while True:
        # Get CPU and GPU usage
        cpu_usage = get_cpu_usage()
        #gpu_usage = get_gpu_usage()
        ram_usage = get_RAM_usage()

        # Add CPU and GPU usage to the cumulative total
        total_cpu_usage += cpu_usage
        #total_gpu_usage += gpu_usage
        total_ram_usage += ram_usage
        count += 1

        # Check if the target script has terminated
        if process.poll() is not None:
            break

        # Sleep for some time before checking again
        time.sleep(1)

    # Calculate average CPU and GPU usage
    average_cpu_usage = total_cpu_usage / count
    average_gpu_usage = total_gpu_usage / count
    average_ram_usage = total_ram_usage / count

    # Print the average CPU and GPU usage at the end
    print(f"Average CPU Usage: {average_cpu_usage}%")
    print(f"Average GPU Usage: {average_gpu_usage}%")
    print(f"Average RAM Usage: {average_ram_usage}%")

    # Calculate total runtime of the target script
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total runtime: {runtime} seconds")
