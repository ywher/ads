import numpy as np
from PIL import Image

def colorize_prediction(prediction_image, color_map):
    """
    color the prediction of semantic segmentation

    参数：
    - prediction_image: PIL Image图像, 预测结果图像的 NumPy 数组，其元素为类别标签。
    - color_map: 字典, 包含类别标签与对应颜色的映射关系

    返回值：
    - colored_image: 上色后的 PIL.Image 对象
    """
    # 将PIL Image转换为NumPy数组
    prediction_array = np.array(prediction_image)

    # 获取图像大小
    height, width = prediction_array.shape

    # 创建一个新的数组来存储上色后的图像
    colored_array = np.zeros((height, width, 3), dtype=np.uint8)

    # 根据颜色映射给预测结果图上色
    for label, color in color_map.items():
        # 找到与类别标签对应的像素点，并将其设为相应的颜色
        colored_array[prediction_array == label] = color

    # 将NumPy数组转换回PIL Image
    colored_image = Image.fromarray(colored_array)

    return colored_image


def colorize_prediction_cv2(prediction_image, color_map):
    """
    color the prediction of semantic segmentation

    args:
    - prediction_image: 预测结果图像的 NumPy 数组, 其元素为类别标签. [H, W]
    - color_map: 字典, 包含类别标签与对应颜色的映射关系. dict[int, tuple]

    return:
    - colored_image: 上色后的 numpy 对象, [H, W, 3]
    """

    # 获取图像大小
    height, width = prediction_image.shape

    # 创建一个新的数组来存储上色后的图像
    colored_array = np.zeros((height, width, 3), dtype=np.uint8)

    # 根据颜色映射给预测结果图上色
    for label, color in color_map.items():
        # 找到与类别标签对应的像素点，并将其设为相应的颜色
        colored_array[prediction_image == label] = color

    return colored_array