import os

def rename_files(folder_path, prefix, start_index):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through the list of files and rename them
    for index, filename in enumerate(file_list):
        # Create a new filename by adding the prefix and the index number
        new_filename = f"{prefix}_{str(start_index + index).zfill(3)}{os.path.splitext(filename)[1]}"
        # Join the folder path with the old and new filenames
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_filename}")

if __name__ == "__main__":
    folder_path = "Archive/2 menit cantika/90-120 detik"  # Replace this with the actual path to your folder
    prefix = "Image"  # Replace this with the desired prefix for your files
    start_index = 451  # Replace this with the starting index value you want
    rename_files(folder_path, prefix, start_index)

# import os

# def rename_content_in_folders(folder_paths, prefix, start_index):
#     for folder_path in folder_paths:
#         # Get a list of all files in the folder
#         file_list = os.listdir(folder_path)

#         # Iterate through the list of files and rename the content
#         for index, filename in enumerate(file_list):
#             # Create a new content by adding the prefix and the index number
#             new_content = f"{prefix}_{start_index + index}"
#             # Join the folder path with the filename
#             file_path = os.path.join(folder_path, filename)

#             # Read the content from the file
#             with open(file_path, 'r') as file:
#                 content = file.read()

#             # Modify the content
#             modified_content = content.replace('old_string', new_content)

#             # Write the modified content back to the file
#             with open(file_path, 'w') as file:
#                 file.write(modified_content)

#             print(f"Renamed content in {file_path}")

# if __name__ == "__main__":
#     folder_paths = ["Archive/sandra 2 menit/30 detik", "Archive/sandra 2 menit/30-60 detik", "Archive/sandra 2 menit/60-90 detik", "Archive/sandra 2 menit/90-120 detik"]
#     # Replace the above list with the actual paths to the folders you want to modify.

#     prefix = "Image"
#     start_index = 1
#     rename_content_in_folders(folder_paths, prefix, start_index)

