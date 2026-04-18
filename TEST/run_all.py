import subprocess
import sys

def run_script(script_name):
    print(f"\n>>> Executing {script_name}...")
    try:
        # Using sys.executable to ensure we use the same environment's python
        result = subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to execute {script_name}: {e}")
        return False

def main():
    print("==========================================")
    print("   BANKRUPTCY PREVENTION PROJECT MASTER   ")
    print("==========================================")
    
    # 1. Run Data Pipeline
    if not run_script('data_pipeline.py'):
        print("Pipeline failed. Aborting.")
        return

    # 2. Run Model Training
    if not run_script('final_bankruptcy_model.py'):
        print("Model training failed. Aborting.")
        return

    print("\n==========================================")
    print("   PROJECT EXECUTION COMPLETE SUCCESS    ")
    print("==========================================")

if __name__ == "__main__":
    main()
