import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_cnt",
        type=int,
        default=1,
        help="Number of images to be loaded",
    )
    parser.add_argument(
        "--upscale_method",
        type=str,
        default="nearest-exact",
        help="Upscale method",
    )
    parser.add_argument(
        "--upscale_width",
        type=int,
        default=512,
        help="Upscale width",
    )
    parser.add_argument(
        "--upscale_height",
        type=int,
        default=512,
        help="Upscale height",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    workflow = {}
    nodes = []
    id_cnt = 0

    # Checkpoint Loader: 0
    nodes.append(
        {
            "id": id_cnt,
            "type": "CheckpointLoaderSimple",
            "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": [], "slot_index": 0},
                {"name": "CLIP", "type": "CLIP", "links": [], "slot_index": 1},
                {"name": "VAE", "type": "VAE", "links": [], "slot_index": 2},
            ],
        }
    )
    id_cnt += 1
    # CLIP Encoder: 1, 2
    nodes.append(
        {
            "id": id_cnt,
            "type": "CLIPTextEncode",
            "inputs": [{"name": "clip", "type": "CLIP", "link": None}],
            "outputs": [
                {
                    "name": "CONDITIONING",
                    "type": "CONDITIONING",
                    "links": [],
                    "slot_index": 0,
                }
            ],
        }
    )
    id_cnt += 1
    nodes.append(
        {
            "id": id_cnt,
            "type": "CLIPTextEncode",
            "inputs": [{"name": "clip", "type": "CLIP", "link": None}],
            "outputs": [
                {
                    "name": "CONDITIONING",
                    "type": "CONDITIONING",
                    "links": [],
                    "slot_index": 0,
                }
            ],
        }
    )
    id_cnt += 1

    for i in range(args.image_cnt):
        # Image Loader: 3 + (2 + 5 * args.upscale_width) * i
        nodes.append(
            {
                "id": id_cnt,
                "type": "LoadImage",
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
                    {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1},
                ],
            }
        )
        id_cnt += 1
        # Image Scaler: 4 + (2 + 5 * args.upscale_width) * i
        nodes.append(
            {
                "id": id_cnt,
                "type": "ImageScale",
                "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0}
                ],
                "widgets_values": [args.upscale_method, args.upscale_width, args.upscale_height, "center"],
            }
        )
        id_cnt += 1

        # Mask In-paints
        for j in range(args.upscale_width):
            # VAE Encoder: 5 + (2 + 5 * args.upscale_width) * i + 5 * j
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "VAEEncodeForInpaint",
                    "inputs": [
                        {"name": "pixels", "type": "IMAGE", "link": None},
                        {"name": "vae", "type": "VAE", "link": None},
                        {"name": "mask", "type": "MASK", "link": None},
                    ],
                    "outputs": [
                        {"name": "LATENT", "type": "LATENT", "links": [], "slot_index": 0}
                    ],
                    "widgets_values": [0],
                }
            )
            id_cnt += 1
            # Image Mask Loader: 6 + (2 + 5 * args.upscale_width) * i + 5 * j
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "LoadImageMask",
                    "outputs": [
                        {"name": "MASK", "type": "MASK", "links": [], "slot_index": 0}
                    ],
                    "widgets_values": [f"mask_{j}.png", "alpha", "image"],
                }
            )
            id_cnt += 1
            # KSampler: 7 + (2 + 5 * args.upscale_width) * i + 5 * j
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "KSampler",
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": None},
                        {"name": "positive", "type": "CONDITIONING", "link": None},
                        {"name": "negative", "type": "CONDITIONING", "link": None},
                        {"name": "latent_image", "type": "LATENT", "link": None},
                    ],
                    "outputs": [
                        {"name": "LATENT", "type": "LATENT", "links": [], "slot_index": 0}
                    ],
                    "widgets_values": [0, "randomize", 64, 0, "euler", "normal", 1],
                }
            )
            id_cnt += 1
            # VAE Decoder: 8 + (2 + 5 * args.upscale_width) * i + 5 * j
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "VAEDecode",
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "link": None},
                        {"name": "vae", "type": "VAE", "link": None},
                    ],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0}
                    ],
                }
            )
            id_cnt += 1
            # Image Saver: 9 + (2 + 5 * args.upscale_width) * i + 5 * j
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "SaveImage",
                    "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                    "widgets_values": [f"image-{i}_mask-{j}"],
                }
            )
            id_cnt += 1

    # === Links ===
    # link id
    # from id
    # from slot
    # to id
    # to slot

    ## Checkpoint Loader -> CLIP Encoder
    links = []
    link_cnt = 0
    nodes[0]["outputs"][1]["links"].append(link_cnt)
    nodes[1]["inputs"][0].update({"link": link_cnt})
    links.append([link_cnt, nodes[0]["id"], 1, nodes[1]["id"], 0])
    link_cnt += 1
    nodes[0]["outputs"][1]["links"].append(link_cnt)
    nodes[2]["inputs"][0].update({"link": link_cnt})
    links.append([link_cnt, nodes[0]["id"], 1, nodes[2]["id"], 0])
    link_cnt += 1

    ## Image Loader -> Image Scaler
    for i in range(args.image_cnt):
        nodes[3 + (2 + 5 * args.upscale_width) * i]["outputs"][0]["links"].append(link_cnt)
        nodes[4 + (2 + 5 * args.upscale_width) * i]["inputs"][0].update({"link": link_cnt})
        links.append([link_cnt, nodes[3 + (2 + 5 * args.upscale_width) * i]["id"], 0, nodes[4 + (2 + 5 * args.upscale_width) * i]["id"], 0])
        link_cnt += 1

        for j in range(args.upscale_width):
            ## Checkpoint -> VAE Encoder
            nodes[0]["outputs"][2]["links"].append(link_cnt)
            nodes[5 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][1].update({"link": link_cnt})
            links.append([link_cnt, nodes[0]["id"], 2, nodes[5 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 1])
            link_cnt += 1

            ## Image Scaler -> VAE Encoder
            nodes[4 + (2 + 5 * args.upscale_width) * i]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][0].update({"link": link_cnt})
            links.append([link_cnt, nodes[4 + (2 + 5 * args.upscale_width) * i]["id"], 0, nodes[5 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 0])
            link_cnt += 1

            ## Image Mask Loader -> VAE Encoder
            nodes[6 + (2 + 5 * args.upscale_width) * i + 5 * j]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][2].update({"link": link_cnt})
            links.append([link_cnt, nodes[6 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 0, nodes[5 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 2])
            link_cnt += 1

            ## Checkpoint -> KSampler
            nodes[0]["outputs"][0]["links"].append(link_cnt)
            nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][0].update({"link": link_cnt})
            links.append([link_cnt, nodes[0]["id"], 0, nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 0])
            link_cnt += 1

            ## CLIP Encoder -> KSampler
            nodes[1]["outputs"][0]["links"].append(link_cnt)
            nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][1].update({"link": link_cnt})
            links.append([link_cnt, nodes[1]["id"], 0, nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 1])
            link_cnt += 1
            nodes[2]["outputs"][0]["links"].append(link_cnt)
            nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][2].update({"link": link_cnt})
            links.append([link_cnt, nodes[2]["id"], 0, nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 2])
            link_cnt += 1

            ## VAE Encoder -> KSampler
            nodes[5 + (2 + 5 * args.upscale_width) * i + 5 * j]["outputs"][0]["links"].append(link_cnt)
            nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][3].update({"link": link_cnt})
            links.append([link_cnt, nodes[5 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 0, nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 3])
            link_cnt += 1

            ## Checkpoint -> VAE Decoder
            nodes[0]["outputs"][0]["links"].append(link_cnt)
            nodes[8 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][1].update({"link": link_cnt})
            links.append([link_cnt, nodes[0]["id"], 2, nodes[9 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 1])
            link_cnt += 1

            ## KSampler -> VAE Decoder
            nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["outputs"][0]["links"].append(link_cnt)
            nodes[8 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][0].update({"link": link_cnt})
            links.append([link_cnt, nodes[7 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 0, nodes[8 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 0])
            link_cnt += 1

            ## VAE Decoder -> Image Saver
            nodes[8 + (2 + 5 * args.upscale_width) * i + 5 * j]["outputs"][0]["links"].append(link_cnt)
            nodes[9 + (2 + 5 * args.upscale_width) * i + 5 * j]["inputs"][0].update({"link": link_cnt})
            links.append([link_cnt, nodes[8 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 0, nodes[9 + (2 + 5 * args.upscale_width) * i + 5 * j]["id"], 0])
            link_cnt += 1

    # dump the workflow
    workflow.update({"nodes": nodes})
    workflow.update({"links": links})
    json.dump(workflow, open("workflow.json", "w"), indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
