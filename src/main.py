import argparse
from utils import *

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the image files"
    )
    return parser.parse_args()

def main(args: argparse.Namespace):
    # Get the folder information
    info = load_folder_information(args.image_path)
    # Generate masks
    origin = load_image(os.path.join(args.image_path, info["origin"]))
    mask_num = len(info["mask_prefix"])
    masks = []
    for i in range(mask_num):
        mask = np.ones_like(origin)
        masked_column = origin.shape[1] // mask_num * (i + 1)
        mask[:, masked_column:] = 0
        masks.append(mask)

    # Image files
    img_paths = os.listdir(args.image_path)
    # MSE
    logits = []
    for i in range(mask_num):
        mask = masks[i]
        mask_prefix = info["mask_prefix"][i]
        imgs = [load_image(os.path.join(args.image_path, img_path)) for img_path in img_paths if mask_prefix in img_path]
        mse = [image_MSE(origin, img, mask) for img in imgs]
        logits.append(np.mean(mse))

    # Get spectrum
    spectrum = sequence_spectrum(logits)
    # Get scores
    scores = spectrum_scores(spectrum)

if __name__ == "__main__":
    args = parse_args()
    main(args)