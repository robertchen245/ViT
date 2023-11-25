import torch

# 创建一个简单的3x3图像
image = torch.tensor([
    [1, 2, 3,4 ],
    [5, 6, 7, 8],
    [9, 10 ,11,12],
    [13,14,15,16]
], dtype=torch.float32).view(1, 1, 3, 3)  # 添加batch和channel维度

# 使用unfold操作划分为2x2的图块
unfolded_image = image.unfold(2, 2, 1)

print("原始图像：")
print(image)
print("划分后的图块：")
print(unfolded_image)