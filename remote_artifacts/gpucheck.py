import torch, torchvision, sklearn, PIL, numpy
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    y = x @ x
print("fp16 matmul ok:", y.dtype, bool(torch.isfinite(y).all()))
print("numpy", numpy.__version__, "| sklearn", sklearn.__version__, "| pillow", PIL.__version__)
