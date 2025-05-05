import zipfile

with zipfile.ZipFile("C://Users\khong\Documents\Python Projects\Drive\Drive\data\great-britain.c_300.kmz", "r") as kmz:
    kmz.extractall("extracted_kml")