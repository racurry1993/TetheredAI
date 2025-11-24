import subprocess
import sys
import os


R_EXECUTABLE_PATH = r"C:\Users\rfo7799\AppData\Local\Programs\R\R-4.5.1\bin\x64\Rscript.exe"
# Define the sequence of scripts to run
# Make sure the paths are correct!
SCRIPTS = [
    {"type": "r", "path": "C:/Users/rfo7799\Desktop/Git/TetheredAI/NFL/PROD/Load_Schedule.R"},
    {"type": "python", "path": "C:/Users/rfo7799\Desktop/Git/TetheredAI/NFL/PROD/oddsapi_player_props.py"},
    {"type": "python", "path": "C:/Users/rfo7799\Desktop/Git/TetheredAI/NFL/PROD/NFL_Player_Yards_Passing_Model_Training.py"},
    {"type": "python", "path": "C:/Users/rfo7799\Desktop/Git/TetheredAI/NFL/PROD/NFL_Player_TDs_Passing_Model_Training.py"},
    {"type": "python", "path": "C:/Users/rfo7799\Desktop/Git/TetheredAI/NFL/PROD/Passing_OU_Preds.py"},
    {"type": "python", "path": "C:/Users/rfo7799\Desktop/Git/TetheredAI/NFL/PROD/PassTDs_OU_Preds.py"},
]

def run_script(script_info):
    """Executes a single script based on its type."""
    script_type = script_info["type"]
    script_path = script_info["path"]
    
    print(f"\n--- Running: {script_path} ({script_type.upper()}) ---")
    
    if script_type == "python":
        # Calls the Python interpreter to run the script
        cmd = [sys.executable, script_path] 
    elif script_type == "r":
        # Calls the Rscript executable to run the R file
        # You may need to change 'Rscript' to the full path if it's not in your PATH
        cmd = [R_EXECUTABLE_PATH, script_path]
    else:
        print(f"ERROR: Unknown script type {script_type}")
        return False
    
    try:
        # Run the command and wait for it to finish
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {script_path} completed.")
        return True
    except subprocess.CalledProcessError as e:
        # If 'check=True' is set, this catches non-zero exit codes (errors)
        print(f"FATAL ERROR in {script_path}: Exited with code {e.returncode}")
        print("\n--- Script Output (STDOUT) ---")
        print(e.stdout)
        print("\n--- Script Errors (STDERR) ---")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"ERROR: Interpreter or script not found. Check paths for {script_path}.")
        return False

def main_pipeline():
    """Manages the sequential execution of all scripts."""
    for script in SCRIPTS:
        if not run_script(script):
            print("\n❌ JOB FAILED: Pipeline stopped due to script error.")
            # Exits the entire job if one script fails
            sys.exit(1) 
            
    print("\n✅ JOB COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main_pipeline()