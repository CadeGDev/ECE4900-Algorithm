import os
import subprocess
from jtop import jtop, JtopException
import sys
import time
import csv
import argparse
import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from algorithm_model import Algorithm, config

# Function to run the target Python script with an integer argument
def run_target_script(script_path, args):
    python_interp = sys.executable
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    command = [python_interp, script_path, "--model", str(args.model), "--image", str(args.image)]
    return subprocess.Popen(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple jtop logger')
    # Standard file to store the logs
    parser.add_argument('--script', action="store", required = True, help = "Target script")
    parser.add_argument('--model', action="store", required = True, help = "Model to be used by script", default=1)
    parser.add_argument('--image', action="store", required = True, help = "Image to be processed by script", default=1)
    parser.add_argument('--file', action="store", dest="file", default="/home/l3harris-clinic/Desktop/Algorithm/Output/log.csv")
    args = parser.parse_args()

    print("Collecting Benchmark Data...")
    print("Saving log on {file}".format(file=args.file))
    print("===================================================================")
    start_time = time.time()

    try:
        with jtop() as jetson:
            # Make csv file and setup csv
            with open(args.file, 'w') as csvfile:
                stats = jetson.stats
                # Initialize cws writer
                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                # Write header
                writer.writeheader()
                # Write first row
                writer.writerow(stats)
                # Begin process
                process = run_target_script(args.script, args)
                # Start loop
                while jetson.ok():
                    stats = jetson.stats
                    # Write row
                    writer.writerow(stats)
                    #print("Log at {time}".format(time=stats['time']))
                    if process.poll() is not None:
                        break

                end_time = time.time()
                runtime = end_time - start_time
                print(f"Process completed after {runtime} seconds")

    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed with CTRL-C")
    except IOError:
        print("I/O error")

#python3 /home/l3harris-clinic/Desktop/Algorithm/ECE4900-Algorithm/project_root/track_usage.py --script /home/l3harris-clinic/Desktop/Algorithm/ECE4900-Algorithm/project_root/inference_testing.py --model /home/l3harris-clinic/Desktop/Algorithm/Trained_Models/TrainingModelV2Cont.pt --image /home/l3harris-clinic/Desktop/Algorithm/Spectrograms/spectrogram.png