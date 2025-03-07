import os
import shutil

def split_data(inputs_dir, test_file_path, test_dir, train_dir):
    """
    Splits files from inputs_dir into test_dir and train_dir based on filenames in test_file_path.

    Args:
        inputs_dir (str): Path to the directory containing input files (e.g., "data/inputs250").
        test_file_path (str): Path to the text file listing filenames for the test set (e.g., "test.txt").
        test_dir (str): Path to the directory where test files will be moved (e.g., "data/test").
        train_dir (str): Path to the directory where train files will be moved (e.g., "data/train").
    """

    if not os.path.exists(inputs_dir):
        print(f"Error: Input directory '{inputs_dir}' does not exist.")
        return

    if not os.path.exists(test_file_path):
        print(f"Error: Test file list '{test_file_path}' does not exist.")
        return

    # Create test and train directories if they don't exist
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    with open(test_file_path, 'r') as f:
        test_filenames = set(line.strip() for line in f)  # Read test filenames into a set for efficient lookup

    for filename in os.listdir(inputs_dir):
        input_filepath = os.path.join(inputs_dir, filename)

        if os.path.isfile(input_filepath): # Ensure we are processing files, not subdirectories
            if filename in test_filenames:
                destination_dir = test_dir
            else:
                destination_dir = train_dir

            destination_filepath = os.path.join(destination_dir, filename)

            try:
                shutil.move(input_filepath, destination_filepath) # Use move to avoid copying and deleting
                print(f"Moved '{filename}' to '{destination_dir}'")
            except Exception as e:
                print(f"Error moving '{filename}': {e}")

    print("File splitting complete.")


if __name__ == "__main__":
    inputs_directory = "data/inputs250"
    test_list_file = "test.txt"
    test_output_directory = "data/test"
    train_output_directory = "data/train"

    split_data(inputs_directory, test_list_file, test_output_directory, train_output_directory)