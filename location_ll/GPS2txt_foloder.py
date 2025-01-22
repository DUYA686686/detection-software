import os
import exifread

def get_photo_location(photo_path):
    with open(photo_path, 'rb') as f:
        tags = exifread.process_file(f)
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat = tags['GPS GPSLatitude']
            lon = tags['GPS GPSLongitude']
            # 转换经纬度格式
            lat_ref = tags['GPS GPSLatitudeRef']
            lon_ref = tags['GPS GPSLongitudeRef']
            lat_value = convert_to_degrees(lat)
            lon_value = convert_to_degrees(lon)
            if lat_ref.values != 'N':
                lat_value = -lat_value
            if lon_ref.values != 'E':
                lon_value = -lon_value
            return lat_value, lon_value
        else:
            return None

def convert_to_degrees(value):
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

# 定义文件夹路径
folder_path = 'Practical condition'
output_path = 'Practical condition_2'  # 输出路径，保存 txt 文件的位置

# 创建输出路径
os.makedirs(output_path, exist_ok=True)

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # 拼接图像文件路径
        photo_path = os.path.join(folder_path, filename)

        # 获取图像的经纬度信息
        location = get_photo_location(photo_path)

        # 创建 txt 文件并保存经纬度信息
        if location:
            txt_file_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.txt")
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(f"{location[0]}, {location[1]}")

print("文档生成完成，保存在指定路径：", output_path)