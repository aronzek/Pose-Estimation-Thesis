import rasterio

tiff_path = "mission_1/data/south_terminal/odm_orthophoto.tif"

with rasterio.open(tiff_path) as dataset:
    print("📌 Orthophoto Metadata:")
    print(f"✅ CRS: {dataset.crs}")
    print(f"✅ Bounds: {dataset.bounds}")  # (minx, miny, maxx, maxy)
    print(f"✅ Resolution: {dataset.res}")  # (pixel width, pixel height)
    print(f"✅ Transform: {dataset.transform}")  # Affine transformation matrix
    print(f"✅ Image Size: {dataset.width} x {dataset.height}")