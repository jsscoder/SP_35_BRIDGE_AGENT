import sam3
print("sam3 ok", getattr(sam3, "__file__", ""))
from sam3.model_builder import build_sam3_image_model
print("builder ok")
