from pathlib import Path
import zipfile 

## -------------unzip the deep fashion dataset------------------------- 
root = Path(r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Data\DeepFashion\Images")

# check how many zip file exist in the directory 
zip_files = list(root.glob("*.zip"))
print(f"found {len(zip_files)}")

for z in root.glob('*.zip'):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(root)
        

## --------------- Unzip the Fashion-gen dataset-------------------------
file_path = Path(r"D:\03. Projects\technocolabs_projects\1. Deep-Learning-Driven-Virtual-Fashion-Fitting\Data\Fashion-gen\FashionGen.zip")

# Define the extraction destination (parent directory)
extract_path = file_path.parent

with zipfile.ZipFile(file_path) as zf:
    zf.extractall(extract_path)