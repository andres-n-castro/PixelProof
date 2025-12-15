import os
from zipfile import ZipFile
import shutil
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="temporarily stores kaggle dataset onto colab disk")
  parser.add_argument("colab_disk_path", metavar="path", type=str, help="argument for the colab disk path")
  args = parser.parse_args()
  colab_path = args.colab_disk_path
  real_folder = os.path.join(colab_path, 'real')
  fake_folder = os.path.join(colab_path, 'fake')
  os.makedirs(real_folder, exist_ok=True)
  os.makedirs(fake_folder, exist_ok=True)
  num_orig_files = 0
  num_deepfake_files = 0

  api = KaggleApi()
  api.authenticate()


  try:
    print("found dataset!")
    api.dataset_download_files(dataset="xdxd003/ff-c23", path=colab_path, unzip=False)
    print("download finished!")

    with ZipFile(file=os.path.join(colab_path, "ff-c23.zip"), mode='r', allowZip64=True) as faceforensics_zip:
      files = faceforensics_zip.namelist()

      for file in files:
        if 'original' in file and file.endswith('.mp4'):
          dest = real_folder
          num_orig_files += 1
                   
        elif 'Deepfakes' in file and file.endswith('.mp4'):
          dest = fake_folder
          num_deepfake_files += 1
        else:
          continue
        
        tot_processed_video_files = num_orig_files + num_deepfake_files
        if tot_processed_video_files % 100 == 0:
          print(f"proccessed {tot_processed_video_files} video files so far...")

        file_name = os.path.basename(file)
        with faceforensics_zip.open(file, 'rb') as source_file, open(os.path.join(dest,file_name), 'wb') as target_file: 
          shutil.copyfileobj(source_file, target_file)
        
    print(f"{num_orig_files} original video files stored | {num_deepfake_files} deepfake video files stored")
    print("Removing downloaded zip file...")
    os.remove(os.path.join(colab_path, 'ff-c23.zip'))
    print("All Done!")

  except Exception as e:
    print(f"error: {e}")