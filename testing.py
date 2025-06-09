import os
import sys

# Print current working directory
print("Current working directory:", os.getcwd())

# Print contents of current directory
print("\nFiles and folders in current directory:")
for item in os.listdir():
    print(" -", item)

# Print sys.path to see where Python is looking for modules
print("\nPython sys.path:")
for path in sys.path:
    print(" -", path)

# Add utils to sys.path if not already included
utils_path = os.path.abspath("utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)
    print(f"\nAdded 'utils' to sys.path: {utils_path}")