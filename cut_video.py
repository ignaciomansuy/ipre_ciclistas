import subprocess
import importlib
import VARIABLES

# Reload the module
importlib.reload(VARIABLES)

# Now you can access the updated variables
from VARIABLES import *

def seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


# Define the command you want to execute
print(SOURCE_ORIGINAL_VIDEO_PATH)
print(SOURCE_VIDEO_PATH)
command = f"ffmpeg -y -i {SOURCE_ORIGINAL_VIDEO_PATH} -ss 00:06:41 -to 00:07:10 -c:v copy -c:a copy {SOURCE_VIDEO_PATH}"  # Example command to list files in the current directory

# Execute the command
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Check if the command was successful
if result.returncode == 0:
    print(result.stdout)
else:
    print("Command failed with error:")
    print(result.stdout)
    print(result.stderr)