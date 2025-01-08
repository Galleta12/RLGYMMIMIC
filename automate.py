import subprocess

# List of commands to execute sequentially
commands = [
    " python imitate3.py --cfg back_flip_demo_seed2 --iter 200 --num_threads 12",
    " python imitate3.py --cfg back_flip_demo_seed3 --iter 200 --num_threads 12",
   
    
]



#export OMP_NUM_THREADS=1

# Execute each command sequentially
for cmd in commands:
    print(f"Running command: {cmd}")
    process = subprocess.run(cmd, shell=True)
    if process.returncode != 0:  # Check for errors
        print(f"Error executing command: {cmd}")
        break
    print(f"Completed: {cmd}\n")
