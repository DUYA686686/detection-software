import piexif


def copy_gps_info(source_image_path, dest_image_path):
    source_exif = piexif.load(source_image_path)
    dest_exif = piexif.load(dest_image_path)

    if "GPS" in source_exif:
        dest_exif["GPS"] = source_exif["GPS"]

    exif_bytes = piexif.dump(dest_exif)
    piexif.insert(exif_bytes, dest_image_path)


copy_gps_info('IMG_20230907_162704.jpg', '9.jpg')
