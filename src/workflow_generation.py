import argparse
import json

GROUP_NODES = {
    "mask": {
        "nodes": [
            {
                "type": "LoadImageMask",
                "outputs": [
                    {"name": "MASK", "type": "MASK", "links": [], "slot_index": 0}
                ],
                "index": 0,
            },
            {
                "type": "VAEEncodeForInpaint",
                "inputs": [
                    {"name": "pixels", "type": "IMAGE", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                    {"name": "mask", "type": "MASK", "link": None},
                ],
                "outputs": [
                    {"name": "LATENT", "type": "LATENT", "links": [], "slot_index": 0}
                ],
                "index": 1,
            },
            {
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
                "index": 2,
            },
            {
                "type": "VAEDecode",
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                ],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0}
                ],
                "index": 3,
            },
            {
                "type": "SaveImage",
                "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                "index": 4,
            },
        ],
        "links": [
            [None, 0, 1, 0, 16, "IMAGE"],
            [None, 2, 1, 1, 4, "VAE"],
            [0, 0, 1, 2, 24, "MASK"],
            [None, 0, 2, 0, 4, "MODEL"],
            [None, 0, 2, 1, 6, "CONDITIONING"],
            [None, 0, 2, 2, 7, "CONDITIONING"],
            [1, 0, 2, 3, 23, "LATENT"],
            [2, 0, 3, 0, 20, "LATENT"],
            [None, 2, 3, 1, 4, "VAE"],
            [3, 0, 4, 0, 21, "IMAGE"],
        ],
        "external": [],
    }
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_cnt",
        type=int,
        default=1,
        help="Number of images to be loaded",
    )
    parser.add_argument(
        "--mask_cnt",
        type=int,
        default=8,
        help="Number of masks to divide image into strips",
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
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="KSampler steps",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    workflow = {}
    nodes = []
    id_cnt = 0

    # === Nodes ===
    # Human images
    # Estimator Checkpoint Loader: 0
    nodes.append(
        {
            "id": id_cnt,
            "type": "CheckpointLoaderSimple",
            "pos": [0, 0],
            "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": [], "slot_index": 0},
                {"name": "CLIP", "type": "CLIP", "links": [], "slot_index": 1},
                {"name": "VAE", "type": "VAE", "links": [], "slot_index": 2},
            ],
        }
    )
    id_cnt += 1
    # Estimator CLIP Encoder: 1, 2
    nodes.append(
        {
            "id": id_cnt,
            "type": "CLIPTextEncode",
            "pos": [400, 0],
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
            "pos": [400, 200],
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
        # Image Loader: 3 + (2 + args.mask_cnt) * i
        nodes.append(
            {
                "id": id_cnt,
                "type": "LoadImage",
                "pos": [1000, 700 * i],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
                    {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1},
                ],
            }
            # TODO: image name
        )
        id_cnt += 1
        # Image Scaler: 4 + (2 + args.mask_cnt) * i
        nodes.append(
            {
                "id": id_cnt,
                "type": "ImageScale",
                "pos": [1000, 700 * i + 350],
                "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0}
                ],
                "widgets_values": [args.upscale_method, args.upscale_width, args.upscale_height, "center"],
            }
        )
        id_cnt += 1

        # Mask In-paints
        for j in range(args.mask_cnt):
            # Mask: 5 + (2 + args.mask_cnt) * i + j
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "workflow/mask",
                    "pos": [1000 + 400 * (j + 1), 700 * i],
                    "inputs": [
                        {"name": "pixels", "type": "IMAGE", "link": None},
                        {"name": "vae", "type": "VAE", "link": None},
                        {"name": "model", "type": "MODEL", "link": None},
                        {"name": "positive", "type": "CONDITIONING", "link": None},
                        {"name": "negative", "type": "CONDITIONING", "link": None},
                        {"name": "VAEDecode vae", "type": "VAE", "link": None},
                    ],
                    "title": f"Image-{i} Mask-{j}",
                    "widgets_values": [
                        f"mask_{j}.png",
                        "alpha",
                        "image",
                        0,
                        0,
                        "randomize",
                        args.steps,
                        0,
                        "euler",
                        "normal",
                        1,
                        f"image-{i}_mask-{j}",
                    ],
                }
            )
            id_cnt += 1

    # Machine images
    # Subject Checkpoint Loader: 3 + (2 + args.mask_cnt) * args.image_cnt
    nodes.append(
        {
            "id": id_cnt,
            "type": "CheckpointLoaderSimple",
            "pos": [0, 200 + 700 * args.image_cnt],
            "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": [], "slot_index": 0},
                {"name": "CLIP", "type": "CLIP", "links": [], "slot_index": 1},
                {"name": "VAE", "type": "VAE", "links": [], "slot_index": 2},
            ],
        }
    )
    id_cnt += 1
    # Subject CLIP Encoder: 4, 5 + (2 + args.mask_cnt) * args.image_cnt
    nodes.append(
        {
            "id": id_cnt,
            "type": "CLIPTextEncode",
            "pos": [400, 200 + 700 * args.image_cnt],
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
            "pos": [400, 400 + 700 * args.image_cnt],
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
        # Image Loader: 6 + (2 + args.mask_cnt) * args.image_cnt + (2 + args.mask_cnt) * i
        nodes.append(
            {
                "id": id_cnt,
                "type": "LoadImage",
                "pos": [1000, 200 + 700 * (args.image_cnt + i)],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
                    {"name": "MASK", "type": "MASK", "links": [], "slot_index": 1},
                ],
            }
            # TODO: image name
        )
        id_cnt += 1
        # Image Scaler: 7 + (2 + args.mask_cnt) * args.image_cnt + (2 + args.mask_cnt) * i
        nodes.append(
            {
                "id": id_cnt,
                "type": "ImageScale",
                "pos": [1000, 550 + 700 * (args.image_cnt + i)],
                "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0}
                ],
                "widgets_values": [
                    args.upscale_method,
                    args.upscale_width,
                    args.upscale_height,
                    "center",
                ],
            }
        )
        id_cnt += 1

        # Mask In-paints
        for j in range(args.mask_cnt):
            # Mask: 7 + (2 + args.mask_cnt) * args.image_cnt + (2 + args.mask_cnt) * i + j
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "workflow/mask",
                    "pos": [1000 + 400 * (j + 1), 200 + 700 * (args.image_cnt + i)],
                    "inputs": [
                        {"name": "pixels", "type": "IMAGE", "link": None},
                        {"name": "vae", "type": "VAE", "link": None},
                        {"name": "model", "type": "MODEL", "link": None},
                        {"name": "positive", "type": "CONDITIONING", "link": None},
                        {"name": "negative", "type": "CONDITIONING", "link": None},
                        {"name": "VAEDecode vae", "type": "VAE", "link": None},
                    ],
                    "title": f"Image-{i} Mask-{j}",
                    "widgets_values": [
                        f"mask_{j}.png",
                        "alpha",
                        "image",
                        0,
                        0,
                        "randomize",
                        args.steps,
                        0,
                        "euler",
                        "normal",
                        1,
                        f"image-{i}_mask-{j}",
                    ],
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
        nodes[3 + (2 + args.mask_cnt) * i]["outputs"][0]["links"].append(link_cnt)
        nodes[4 + (2 + args.mask_cnt) * i]["inputs"][0].update({"link": link_cnt})
        links.append([link_cnt, nodes[3 + (2 + args.mask_cnt) * i]["id"], 0, nodes[4 + (2 + args.mask_cnt) * i]["id"], 0])
        link_cnt += 1

        for j in range(args.mask_cnt):
            ## Checkpoint -> Mask-KSampler
            nodes[0]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + (2 + args.mask_cnt) * i + j]["inputs"][2].update({"link": link_cnt})
            links.append([link_cnt, nodes[0]["id"], 0, nodes[5 + (2 + args.mask_cnt) * i + j]["id"], 2])
            link_cnt += 1
            ## Checkpoint -> Mask-VAE Encoder
            nodes[0]["outputs"][2]["links"].append(link_cnt)
            nodes[5 + (2 + args.mask_cnt) * i + j]["inputs"][1].update({"link": link_cnt})
            links.append([link_cnt, nodes[0]["id"], 2, nodes[5 + (2 + args.mask_cnt) * i + j]["id"], 1])
            link_cnt += 1
            ## Checkpoint -> Mask-VAE Decoder
            nodes[0]["outputs"][2]["links"].append(link_cnt)
            nodes[5 + (2 + args.mask_cnt) * i + j]["inputs"][5].update({"link": link_cnt})
            links.append([link_cnt, nodes[0]["id"], 2, nodes[5 + (2 + args.mask_cnt) * i + j]["id"], 5])
            link_cnt += 1

            ## CLIP Encoder -> Mask-KSampler
            nodes[1]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + (2 + args.mask_cnt) * i + j]["inputs"][3].update({"link": link_cnt})
            links.append([link_cnt, nodes[1]["id"], 0, nodes[5 + (2 + args.mask_cnt) * i + j]["id"], 3])
            link_cnt += 1
            nodes[2]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + (2 + args.mask_cnt) * i + j]["inputs"][4].update({"link": link_cnt})
            links.append([link_cnt, nodes[2]["id"], 0, nodes[5 + (2 + args.mask_cnt) * i + j]["id"], 4])
            link_cnt += 1

            ## Image Scaler -> Mask-VAE Encoder
            nodes[4 + (2 + args.mask_cnt) * i]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + (2 + args.mask_cnt) * i + j]["inputs"][0].update({"link": link_cnt})
            links.append([link_cnt, nodes[4 + (2 + args.mask_cnt) * i]["id"], 0, nodes[5 + (2 + args.mask_cnt) * i + j]["id"], 0])
            link_cnt += 1

    # dump the workflow
    workflow.update({"nodes": nodes})
    workflow.update({"links": links})
    workflow.update({"extra": {"groupNodes": GROUP_NODES}})
    json.dump(workflow, open("workflow.json", "w"), indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
