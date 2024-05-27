import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import json
import warnings

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

checkpoint = "sam_vit_h_4b8939.pth"
# checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=checkpoint)

# onnx_model_path = None  # Set to use an already exported model, then skip to the next section.

onnx_model_path = "sam_onnx_example.onnx"

onnx_model = SamOnnxModel(sam, return_single_mask=True)

dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}

embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
}
output_names = ["masks", "iou_predictions", "low_res_masks"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

'''
onnx_model_quantized_path = "sam_onnx_quantized_example.onnx"
quantize_dynamic(
    model_input=onnx_model_path,
    model_output=onnx_model_quantized_path,
    optimize_model=True,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QUInt8,
)
onnx_model_path = onnx_model_quantized_path
'''

ort_session = onnxruntime.InferenceSession(onnx_model_path)

# sam.to(device='cuda')
predictor = SamPredictor(sam)

while True:
  print('-file')
  image_path = input()
  if image_path == '-q':
    break
  image = cv2.imread(image_path)
  print(f"size: {image.shape}", file=sys.stderr)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  predictor.set_image(image)

  image_embedding = predictor.get_image_embedding().cpu().numpy()

  # np.save('image_embedding.npy', image_embedding)

  print("-loaded")

  while True:
    point_str = input()
    if point_str == '-q':
      break
    point_info = json.loads(point_str)

    input_point = np.array(point_info)
    input_label = np.array([1] * len(point_info))

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold

    h, w = masks.shape[-2:]
    print(f"sizp: {h} {w}", file=sys.stderr)

    print(masks.shape, file=sys.stderr)
    y_coords, x_coords = np.where(masks.reshape(h, w))

    if len(x_coords) == 0:
      print(f'-result [0, 0, 0, 0]')
    else:
      x_min, x_max = x_coords.min(), x_coords.max()
      y_min, y_max = y_coords.min(), y_coords.max()

      print(f'-result [{x_min}, {y_min}, {x_max}, {y_max}]')

    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_mask(masks, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # show_box([x_min, y_min, x_max, y_max], plt.gca())
    # plt.axis('off')
    # plt.show()