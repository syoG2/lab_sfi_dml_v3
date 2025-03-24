import glob
import multiprocessing


def count_lines_in_file(filepath):
    with open(filepath, "r") as file:
        return sum(1 for line in file)


def count_lines_in_files(directory_pattern):
    filepaths = glob.glob(directory_pattern)
    with multiprocessing.Pool() as pool:
        total_lines = sum(pool.map(count_lines_in_file, filepaths))
    return total_lines


if __name__ == "__main__":
    directory_pattern = "./data/preprocessing/c4/preprocess/token0/train_*/lu/*.jsonl"
    total_lines = count_lines_in_files(directory_pattern)
    print(f"Total lines: {total_lines}")
