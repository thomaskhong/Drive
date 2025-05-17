import zipfile

with zipfile.ZipFile("data\great-britain.c_300.kmz", "r") as kmz:
    kmz.extractall("extracted_kml")