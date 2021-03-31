import gdown
import zipfile

url = "https://drive.google.com/u/1/uc?id=1-GUAGJcSoiBaR8ecylUlJPvDJtqxCDLE&export=download"
output = "data.zip"
gdown.download(url, output)

with zipfile.ZipFile("data.zip", 'r') as zip_ref:
    zip_ref.extractall("./")


