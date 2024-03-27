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

    # Checkpoint Loader
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
    # CLIP Encoder
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
        # Image Loader
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
        # Image Scaler
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
        nodes[3 + i * 2]["outputs"][0]["links"].append(link_cnt)
        nodes[4 + i * 2]["inputs"][0].update({"link": link_cnt})
        links.append([link_cnt, nodes[3 + i * 2]["id"], 0, nodes[4 + i * 2]["id"], 0])
        link_cnt += 1

    # dump the workflow
    workflow.update({"nodes": nodes})
    workflow.update({"links": links})
    json.dump(workflow, open("workflow.json", "w"), indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
