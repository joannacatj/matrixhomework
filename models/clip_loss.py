import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def calculate_clip_loss(image_list, text_list):
    assert len(image_list) == len(text_list), "length not same!"

    text_inputs = [clip.tokenize(text) for text in text_list]
    # print(text_inputs[0].shape)
    text_inputs = torch.cat(text_inputs, dim=0).to(device)
    # print(text_inputs.shape)

    # image_inputs = torch.stack([preprocess(Image.open(image)).unsqueeze(0) for image in image_list]).to(device)
    image_tensor_list = [preprocess(image).unsqueeze(0) for image in image_list]
    image_inputs = torch.cat(image_tensor_list, dim=0).to(device)
    # print(image_inputs.shape)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)

    similarities = (image_features @ text_features.t()).diagonal()
    clip_loss = (1. - similarities / 100)
    #print(clip_loss)

    return clip_loss
