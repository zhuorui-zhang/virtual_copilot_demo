# File path for the input file
input_file_path = 'character'  # Replace with your actual input file path

# File path for the output file
output_file_path = '../test/wav.scp'

# Directory path for .wav files
wav_directory_path = 'F:\\\ATCandPilotAudio\\\subtitle\\'

# Open the input file and read lines
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Open the output file for writing
with open(output_file_path, 'w', encoding='utf-8') as file:
    for line in lines:
        # Extract the .wav file name from each line
        wav_name = line.split()[0]  # Assuming the .wav file name is the first word in each line
        # Create the new line format
        new_line = f'{wav_name} {wav_directory_path}\\{wav_name}\n'
        # Write to the output file
        file.write(new_line)

print(f"File '{output_file_path}' has been created with the formatted paths.")
