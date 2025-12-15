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
    page_token = None
    files_list_result = api.dataset_list_files("xdxd003/ff-c23", page_token=page_token, page_size=200)
    files = files_list_result.files

    for file in files:
      if "original" in file.name and file.name.endswith(".mp4"):
        print(f"found original video file : {file.name}")
        real_path = os.path.join(args.colab_disk_path, "real")
        os.makedirs(real_path,exist_ok=True)
        api.dataset_download_file(dataset="xdxd003/ff-c23", file_name=file.name, path=real_path)
      elif "Deepfakes" in file.name and file.name.endswith(".mp4"):
        print(f"found Deepfake video file : {file.name}")
        fake_path = os.path.join(args.colab_disk_path, "fake")
        os.makedirs(fake_path,exist_ok=True)
        api.dataset_download_file(dataset="xdxd003/ff-c23", file_name=file.name, path=fake_path)
      else:
        continue
  except Exception as e:
    print(f"error: {e}")