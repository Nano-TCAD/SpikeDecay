import numpy as np
import os


directory = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":

    # filenames = ["runtimes_64.txt", "runtimes_1024.txt"]
    # filenames = ["runtimes_fp16_64.txt", "runtimes_fp16_1024.txt"]
    filenames = ["runtimes_bfp16_64.txt", "runtimes_bfp16_1024.txt"]

    for filename in filenames:

        f = os.path.join(directory, filename)
        if os.path.isfile(f):

            with open(f) as file:

                lines = file.readlines()
                weights = []
                activation = []
                decay = []
                stp = []

                for line in lines:
                    if "Total weights time" in line:
                        weights.append(float(line.split(": ")[1]))
                    elif "Activation time" in line:
                        activation.append(float(line.split(": ")[1]))
                    elif "Decay time" in line:
                        decay.append(float(line.split(": ")[1]))
                    elif "STP time" in line:
                        stp.append(float(line.split(": ")[1]))
            
            factor = 1
            print(f"File: {filename}")
            print(f"Total weights time: {1e9 * np.median(weights) / factor} ns")
            print(f"Activation time: {1e9 * np.median(activation) / factor} ns")
            print(f"Decay time: {1e9 * np.median(decay)/ factor} ns")
            print(f"STP time: {1e9 * np.median(stp)/ factor} ns")
            print()
