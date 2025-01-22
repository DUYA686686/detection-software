from PIL import Image
import os

input_path = "Practical condition"  # 图片所在的文件夹路径
output_path = "Practical condition_2"  # 调整后的图片保存路径

# 创建保存路径
os.makedirs(output_path, exist_ok=True)

# 遍历文件夹中的图片
for maindir, subdir, file_name_list in os.walk(input_path):
    for file_name in file_name_list:
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(maindir, file_name)  # 获取每张图片的路径
            output_image_path = os.path.join(output_path, file_name)  # 构建输出路径

            # 打开并调整图片尺寸
            img = Image.open(image_path)
            resized_img = img.resize((227, 227), Image.ANTIALIAS)

            # 保存调整后的图片
            resized_img.save(output_image_path)

print("图片调整完成，保存在指定路径：", output_path)

# from PIL import Image
# import os
#
# input_path = "Practical condition"  # 图片所在的文件夹路径
# output_path = "Practical condition_2"  # 调整后的图片保存路径
#
# # 创建保存路径
# os.makedirs(output_path, exist_ok=True)
#
# # 遍历文件夹中的图片
# for maindir, subdir, file_name_list in os.walk(input_path):
#     for file_name in file_name_list:
#         image_path = os.path.join(maindir, file_name)  # 获取每张图片的路径
#         output_image_path = os.path.join(output_path, file_name)  # 构建输出路径
#
#         # 打开并调整图片尺寸
#         img = Image.open(image_path)
#         resized_img = img.resize((227, 227), Image.ANTIALIAS)
#
#         # 保存调整后的图片
#         resized_img.save(output_image_path)
#
# print("图片调整完成，保存在指定路径：", output_path)