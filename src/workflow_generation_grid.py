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
        default=8,
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

    # === Nodes ===
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
            "pos": [400, 250],
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

    # Batch Image Loader: 3
    nodes.append(
        {
            "id": id_cnt,
            "type": "Load Image Batch",
            "pos": [400, 500],
            "outputs": [
                {"name": "image", "type": "IMAGE", "links": [], "slot_index": 0},
                {"name": "filename_text", "type": "STRING",
                "links": [], "slot_index": 1}
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

    for i in range(args.grid_side):
        for j in range(args.grid_side):
            # Filename String Function: 4 + i * grid_side * 2 + j * 2
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "StringFunction|pysssss",
                    "pos": [970 + 400 * j * 2, 700 * i],
                    "inputs": [
                        {"name": "text_a", "type": "STRING", "link": None, "widget": {"name": "text_a"}}
                    ],
                    "outputs": [
                        {"name": "STRING", "type": "STRING", "links": [], "slot_index": 0}
                    ],
                    "widgets_values": [
                        "append", "yes", "", f"_mask-{i*args.grid_side+j}", ""
                    ]
                }
            )
            id_cnt += 1
            # Masked In-paint: 5 + i * grid_side * 2 + j * 2
            nodes.append(
                {
                    "id": id_cnt,
                    "type": "workflow/MaskInpaint",
                    "pos": [1000 + 400 * (j * 2 + 1), 700 * i],
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

    # === Links ===
    # link id
    # from id
    # from slot
    # to id
    # to slot
    links = []
    link_cnt = 0

    ## Estimator Checkpoint Loader -> CLIP Encoder
    nodes[0]["outputs"][1]["links"].append(link_cnt)
    nodes[1]["inputs"][0].update({"link": link_cnt})
    links.append([link_cnt, nodes[0]["id"], 1, nodes[1]["id"], 0])
    link_cnt += 1
    nodes[0]["outputs"][1]["links"].append(link_cnt)
    nodes[2]["inputs"][0].update({"link": link_cnt})
    links.append([link_cnt, nodes[0]["id"], 1, nodes[2]["id"], 0])
    link_cnt += 1

    for i in range(args.grid_side):
        for j in range(args.grid_side):
            ## String Function -> Masked In-paint
            nodes[4 + i * args.grid_side * 2 + j * 2]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + i * args.grid_side * 2 + j * 2]["inputs"][6].update({"link": link_cnt})
            links.append([link_cnt, nodes[4 + i * args.grid_side * 2 + j * 2]["id"], 0, nodes[5 + i * args.grid_side * 2 + j * 2]["id"], 6])
            link_cnt += 1
            ## Checkpoint Model -> Masked In-paint
            nodes[0]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + i * args.grid_side * 2 + j * 2]["inputs"][2].update({"link": link_cnt})
            links.append([link_cnt, nodes[0]["id"], 0, nodes[5 + i * args.grid_side * 2 + j * 2]["id"], 2])
            link_cnt += 1
            ## Checkpoint VAE -> Masked In-paint
            nodes[0]["outputs"][2]["links"].append(link_cnt)
            nodes[5 + i * args.grid_side * 2 + j * 2]["inputs"][1].update({"link": link_cnt})
            links.append([link_cnt, nodes[0]["id"], 2, nodes[5 + i * args.grid_side * 2 + j * 2]["id"], 1])
            link_cnt += 1
            nodes[0]["outputs"][2]["links"].append(link_cnt)
            nodes[5 + i * args.grid_side * 2 + j * 2]["inputs"][5].update({"link": link_cnt})
            links.append([link_cnt, nodes[0]["id"], 2, nodes[5 + i * args.grid_side * 2 + j * 2]["id"], 5])
            link_cnt += 1
            ## CLIP -> Masked In-paint
            nodes[1]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + i * args.grid_side * 2 + j * 2]["inputs"][3].update({"link": link_cnt})
            links.append([link_cnt, nodes[1]["id"], 0, nodes[5 + i * args.grid_side * 2 + j * 2]["id"], 3])
            link_cnt += 1
            nodes[2]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + i * args.grid_side * 2 + j * 2]["inputs"][4].update({"link": link_cnt})
            links.append([link_cnt, nodes[2]["id"], 0, nodes[5 + i * args.grid_side * 2 + j * 2]["id"], 4])
            link_cnt += 1
            ## Batch Image Loader -> Masked In-paint
            nodes[3]["outputs"][0]["links"].append(link_cnt)
            nodes[5 + i * args.grid_side * 2 + j * 2]["inputs"][0].update({"link": link_cnt})
            links.append([link_cnt, nodes[3]["id"], 0, nodes[5 + i * args.grid_side * 2 + j * 2]["id"], 0])
            link_cnt += 1
            ## Batch Image Loader -> String Function
            nodes[3]["outputs"][1]["links"].append(link_cnt)
            nodes[4 + i * args.grid_side * 2 + j * 2]["inputs"][0].update({"link": link_cnt})
            links.append([link_cnt, nodes[3]["id"], 1, nodes[4 + i * args.grid_side * 2 + j * 2]["id"], 0])
            link_cnt += 1


    # dump the workflow
    workflow.update({"last_node_id": id_cnt})
    workflow.update({"last_link_id": link_cnt})
    workflow.update({"nodes": nodes})
    workflow.update({"links": links})
    workflow.update({"extra": {"groupNodes": GROUP_NODES}})
    json.dump(workflow, open("workflow.json", "w"), indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)
