import os
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="temporarily stores kaggle dataset onto colab disk")
  parser.add_argument("colab_disk_path", metavar="path", type=str, help="argument for the colab disk path")
  args = parser.parse_args()

  api = KaggleApi()
  api.authenticate()

  try:
    files_list_result = api.dataset_list_files("xdxd003/ff-c23")
    files = files_list_result.files

    for file in files:
      if "original" in file.name and file.name.endswith(".mp4"):
        print("found original video file!")
        real_path = os.path.join(args.colab_disk_path, "real")
        os.makedirs(real_path,exist_ok=True)
        api.dataset_download_file(dataset="xdxd003/ff-c23", file_name=file.name, path=real_path)
      elif "Deepfakes" in file.name and file.name.endswith(".mp4"):
        print("found DeepFake video file!")
        fake_path = os.path.join(args.colab_disk_path, "fake")
        os.makedirs(fake_path,exist_ok=True)
        api.dataset_download_file(dataset="xdxd003/ff-c23", file_name=file.name, path=fake_path)
      else:
        continue
  except Exception as e:
    print(f"error: {e}")