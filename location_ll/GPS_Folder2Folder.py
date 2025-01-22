import os
import piexif
import glob

def copy_gps_info(source_folder, dest_folder):
    image_files = glob.glob(os.path.join(source_folder, "*.jpg"))  # 修改文件类型以适应您的需求

    for image_file in image_files:
        image_name = os.path.basename(image_file)
        dest_image_file = os.path.join(dest_folder, image_name)

        if os.path.exists(dest_image_file):
            source_exif = piexif.load(image_file)
            dest_exif = piexif.load(dest_image_file)

            if "GPS" in source_exif:
                dest_exif["GPS"] = source_exif["GPS"]

            exif_bytes = piexif.dump(dest_exif)
            piexif.insert(exif_bytes, dest_image_file)
            print(f"GPS info copied from {image_file} to {dest_image_file}")
        else:
            print(f"Destination image file {dest_image_file} does not exist.")

    print("GPS info copying completed.")

# 调用函数进行测试
copy_gps_info("Practical condition", "Practical condition_2")