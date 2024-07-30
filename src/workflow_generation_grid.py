import argparse
import json
from group_nodes import GROUP_NODES

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_path",
        type=str,
        # required=True,
        help="Path to the batch images",
    )
    parser.add_argument(
        "--grid_side",
        type=int,
        default=2,
        help="Number of grid on one side",
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

    nodes.append(
        {
            "id": id_cnt,
            "type": "Load Image Batch",
            "pos": [400, 0],
            "outputs": [
                {"name": "image", "type": "IMAGE", "links": None, "slot_index": 0},
                {"name": "filename_text", "type": "STRING",
                "links": None, "slot_index": 1}
            ],
            "widgets_values": [
                "incremental_image",
                0,
                "Batch 001",
                args.batch_path,
                "*",
                "true",
                "false"
            ]
        },
    )
    id_cnt += 1

    # Masked In-paint
    for i in range(args.grid_side):
        for j in range(args.grid_side):
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "workflow/MaskInpaint",
                    "pos": [1000 + 400 * (j + 1), 700 * i],
                    "inputs": [
                        {"name": "pixels", "type": "IMAGE", "link": None},
                        {"name": "vae", "type": "VAE", "link": None},
                        {"name": "model", "type": "MODEL", "link": None},
                        {"name": "positive", "type": "CONDITIONING", "link": None},
                        {"name": "negative", "type": "CONDITIONING", "link": None},
                        {"name": "VAEDecode vae", "type": "VAE", "link": None},
                        {"name": "filename_prefix", "type": "STRING", "link": None, "widget": {"name": "filename_prefix"}},
                    ],
                    "title": f"Image-{i} Mask-{j}",
                    "widgets_values": [
                        f"mask_{i*args.grid_side+j}.png",
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
                        f"mask-{i*args.grid_side+j}",
                    ],
                }
            )
            id_cnt += 1

    # dump the workflow
    workflow.update({"last_node_id": id_cnt})
    # workflow.update({"last_link_id": link_cnt})
    workflow.update({"nodes": nodes})
    # workflow.update({"links": links})
    workflow.update({"extra": {"groupNodes": GROUP_NODES}})
    json.dump(workflow, open("workflow.json", "w"), indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)
