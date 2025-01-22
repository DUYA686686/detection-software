from PIL import Image

def read_exif_data(image_path):
    with Image.open(image_path) as img:
        exif_data = img._getexif()
        if exif_data is not None:
            for tag, value in exif_data.items():
                print(f'Tag: {tag}, Value: {value}')
        else:
            print('No exif data found.')

image_path = '1.jpg'  # 图像路径
read_exif_data(image_path)

# 'Practical condition/1.jpg'
# Tag: 34853, Value: {1: 'N', 2: (31.0, 55.0, 0.3972), 3: 'E', 4: (118.0, 46.0, 56.2008), 5: b'\x00', 6: 0.0, 7: (9.0, 2.0, 56.0), 12: 'K', 13: 0.0, 27: 'network', 29: '2023:07:21'}
# 'Practical condition_2/1.jpg'
# Tag: 34853, Value: {0: b'\x02\x03\x00\x00', 2: (0.0, 0.0, 0.0), 4: (0.0, 0.0, 0.0)}