import torch

# สร้าง tensor สำหรับตัวอย่าง
batch_size = 2
num_items = 3
embedding_dim = 4

# สร้าง input tensors
b2 = torch.randn(batch_size, num_items, 1)  # bias
w2 = torch.randn(batch_size, num_items, embedding_dim)  # weights
x = torch.randn(batch_size, embedding_dim)  # input

print("b2 shape:", b2.shape)
print("w2 shape:", w2.shape)
print("x shape:", x.shape)

# ทำการ unsqueeze x เพื่อให้สามารถคูณกับ w2 ได้
x_unsqueezed = x.unsqueeze(2)
print("x_unsqueezed shape:", x_unsqueezed.shape)

# ใช้ torch.baddbmm
result = torch.baddbmm(b2, w2, x_unsqueezed)
print("Result shape:", result.shape)

# ทำการ squeeze เพื่อลบมิติที่มีขนาด 1 ออก
result_squeezed = result.squeeze()
print("Result squeezed shape:", result_squeezed.shape)

# แสดงผลลัพธ์
print("\nFinal result:")
print(result_squeezed)

# เปรียบเทียบกับการคำนวณแบบปกติ
normal_result = (w2 @ x_unsqueezed) + b2
print("\nResult using normal matrix multiplication:")
print(normal_result.squeeze())

# ตรวจสอบว่าผลลัพธ์เท่ากันหรือไม่
print("\nResults are equal:", torch.allclose(result, normal_result))