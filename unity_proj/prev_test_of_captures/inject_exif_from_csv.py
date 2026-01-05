# File: inject_exif_from_csv.py
# Requires: pip install piexif pandas

import piexif
import pandas as pd
import os
from PIL import Image

def dms_coordinates(decimal):
    degrees = int(decimal)
    minutes_float = abs(decimal - degrees) * 60
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60 * 10000)
    return ((abs(degrees), 1), (minutes, 1), (seconds, 10000))

def inject_exif(image_path, lat, lon, alt):
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = dms_coordinates(lat)
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = dms_coordinates(lon)
    exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 0
    exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = (int(alt * 100), 100)

    exif_bytes = piexif.dump(exif_dict)
    img = Image.open(image_path)
    img.save(image_path, exif=exif_bytes)

def main():
    csv_path = "gps_info.csv"
    image_dir = "images"

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        img_file = os.path.join(image_dir, row['filename'])
        lat, lon, alt = float(row['latitude']), float(row['longitude']), float(row['altitude'])

        if os.path.exists(img_file):
            inject_exif(img_file, lat, lon, alt)
            print(f"✅ EXIF injected: {img_file}")
        else:
            print(f"⚠️ File missing: {img_file}")

if __name__ == '__main__':
    main()
