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


# 使用示例
photo_path = '20230907/IMG_20230907_162837.jpg'
location = get_photo_location(photo_path)
if location:
    print('照片拍摄地点：', location)
else:
    print('无法提取照片拍摄地点')
