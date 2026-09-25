import sys
from datasets import Dataset, Image as HFImage
from PIL import Image, ImageDraw
imgs, texts = [], []
for i in range(40):
    im = Image.new("RGB", (224, 64), "white"); ImageDraw.Draw(im).text((8, 24), f"x^{i} + {i}y = {i*3}", fill="black")
    imgs.append(im); texts.append(f"x^{i} + {i}y = {i*3}")
Dataset.from_dict({"image": imgs, "text": texts}).cast_column("image", HFImage()).to_parquet(sys.argv[1])
