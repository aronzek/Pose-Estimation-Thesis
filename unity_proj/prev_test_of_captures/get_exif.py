from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import sys

def get_exif(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()

    if not exif_data:
        print("No EXIF data found.")
        return

    print(f"\nEXIF info for: {image_path}\n")

    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo":
            gps_data = {}
            for t in value:
                sub_tag = GPSTAGS.get(t, t)
                gps_data[sub_tag] = value[t]
            print(f"{tag}: {gps_data}")
        else:
            print(f"{tag}: {value}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_exif.py <image.jpg>")
    else:
        get_exif(sys.argv[1])
