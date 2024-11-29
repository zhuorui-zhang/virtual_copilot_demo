import os

# Path to the directory containing the audio files
directory_path = 'F:/ATCandPilotAudio/Pilot1'

# Path to the file containing the list of audio files with paths
file_a_path = '../test/character.txt'
# Read the list of files from the directory
files_in_directory = set(os.listdir(directory_path))

# Initialize a list to hold the lines that should be kept in File A
lines_to_keep = []

# Open File A, read lines, and check if the corresponding files exist in the directory
with open(file_a_path, 'r', encoding='utf-8') as file:
    for line in file:
        filename = line.split()[0]  # Assumes filename is the first part of the line
        if filename in files_in_directory:
            lines_to_keep.append(line)

# Rewrite File A with only the entries that correspond to files in the directory
with open(file_a_path, 'w', encoding='utf-8') as file:
    file.writelines(lines_to_keep)

print(f"Updated {file_a_path} to only include files that exist in {directory_path}.")