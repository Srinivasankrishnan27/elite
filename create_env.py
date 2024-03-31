import subprocess
import sys
import os

def create_environment(environment_name, environment_file):
    # Check if environment exists
    env_exists = False
    try:
        env_list = subprocess.check_output(["conda", "env", "list"], universal_newlines=True)
        if environment_name in env_list:
            env_exists = True
            print(f"Environment '{environment_name}' already exists.")
    except subprocess.CalledProcessError:
        pass
    
    # Create, update, or delete environment based on existence
    if not env_exists:
        print(f"Creating new environment '{environment_name}'...")
        create_new_environment(environment_name, environment_file)
    else:
        # Check if the environment needs to be updated
        update_environment(environment_name, environment_file)

def create_new_environment(environment_name, environment_file):
    try:
        subprocess.run(["conda", "env", "create", "-n", environment_name, "-f", environment_file], check=True)
        print(f"Environment '{environment_name}' created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error creating environment '{environment_name}': {e}")
        sys.exit(1)

def update_environment(environment_name, environment_file):
    try:
        subprocess.run(["conda", "env", "update", "-n", environment_name, "-f", environment_file], check=True)
        print(f"Environment '{environment_name}' updated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error updating environment '{environment_name}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_env.py <environment_name> <environment.yml>")
        sys.exit(1)
    environment_name = sys.argv[1]
    environment_file = sys.argv[2]
    create_environment(environment_name, environment_file)