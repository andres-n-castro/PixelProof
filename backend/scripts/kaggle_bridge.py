import os
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import pandas as pd
from google.cloud import storage

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="temporarily stores kaggle dataset onto colab disk")
  parser.add_argument("kaggle_input_path", metavar="kaggle input path", type=str, help="argument for the kaggle input directory")
  parser.add_argument("bucket_name", metavar="cloud bucket name", type=str, help="argument for the google cloud bucket name")
  args = parser.parse_args()
  input_path = args.kaggle_input_path
  bucket_name = args.bucket_name

  try:
    storage_client = storage.Client(project='pixelproof-data-storage')
    bucket = storage_client.bucket(bucket_name=bucket_name)

    orig_df = pd.read_csv(os.path.join(input_path, 'original.csv'))
    deepfake_df = pd.read_csv(os.path.join(input_path, 'Deepfakes.csv'))
    master_df = pd.concat([orig_df, deepfake_df], ignore_index=True)

    orig_df_str = orig_df.to_csv(index=False)
    deepfake_df_str = deepfake_df.to_csv(index=False)
    master_df_str = master_df.to_csv(index=False)
    
    with ZipFile(file='csv.zip', mode='w', compression=ZIP_DEFLATED) as csv_zip:
      csv_zip.writestr('original.csv', orig_df_str)
      csv_zip.writestr('Deepfakes.csv', deepfake_df_str)
      csv_zip.writestr('master.csv', master_df_str)
    
    blob_dest = 'data_csv_files/csv.zip'
    blob = bucket.blob(blob_dest)

    print("uploading csv.zip....")
    blob.upload_from_filename('csv.zip')
    print("Success!")

  except Exception as e:
    print(f"error: {e}")