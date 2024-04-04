import os
import shutil
import argparse


def move(algo):
    # Path of the folder containing all the folders to be moved
    os.chdir("results1")

    # Get all the folder names in the one folder 
    folders = os.listdir()

    # Print the folders
    print(folders)

    # Remove the .DS_Store and .ipynb_checkpoint if they exists
    if ".DS_Store" in folders:
        folders.remove(".DS_Store")

    if ".ipynb_checkpoints" in folders:
        folders.remove(".ipynb_checkpoints")



    # Change back to the parent directory
    os.chdir("..")

    # Path of destination folder
    destination = "results"

    # iterate through all the folders
    for folder in folders:
        # Path of the folder to be moved
        src_path = os.path.join("results1", folder, '3', algo)

        # Check if the folder exists in the final path
        des_path = os.path.join(destination, folder, '3', algo)



        # Final path of the folder to be moved
        final_path = os.path.join(destination, folder, '3')
        
        if os.path.exists(des_path):
            print(des_path + " Exists. Replacing...")
            shutil.rmtree(des_path)
        else:
            print(des_path + " Doesn't exist")

        # Move a folder form one location to another
        shutil.move(src_path, final_path)


    shutil.rmtree("results1")

def check(algo):
    # Path of the folder containing all the folders to be checked
    os.chdir("results")

    # Get all the folder names in the one folder 
    folders = os.listdir()

    # Remove the .DS_Store and .ipynb_checkpoint if they exists
    if ".DS_Store" in folders:
        folders.remove(".DS_Store")

    if ".ipynb_checkpoints" in folders:
        folders.remove(".ipynb_checkpoints")

    # Change back to the parent directory
    os.chdir("..")

    # Save folder that don't exist
    not_exist = []

    # iterate through all the folders
    for folder in folders:
        # Path of the folder to be moved
        src_path = os.path.join("results", folder, '3', algo)
        
        # Check if the folder exists
        if os.path.exists(src_path):
            print(folder + " Exists")
        else:
            not_exist.append(folder)
            print(folder + " Doesn't exist")
    
    print("Folders that don't exist: ", not_exist)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument("--algo", type=str, help="The algorithm to be used")
    parser.add_argument("--mode", type=str, help="Choose the mode")

    # Parse the arguments
    args = parser.parse_args()

    # choose the mode
    if args.mode == "move":
        move(args.algo)
    elif args.mode == "check":
        check(args.algo)