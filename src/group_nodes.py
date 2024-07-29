GROUP_NODES = {
    "MaskInpaint": {
        "nodes": [
            {
                "type": "LoadImageMask",
                "outputs": [
                    {"name": "MASK", "type": "MASK", "links": [], "slot_index": 0}
                ],
                "pos": [0, 0],
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
                "pos": [0, 0],
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
                "pos": [0, 0],
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
                "pos": [0, 0],
                "index": 3,
            },
            {
                "type": "SaveImage",
                "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                "pos": [0, 0],
                "index": 4,
            },
        ],
        "links": [
            [0, 0, 1, 2, None, "MASK"],
            [1, 0, 2, 3, None, "LATENT"],
            [2, 0, 3, 0, None, "LATENT"],
            [3, 0, 4, 0, None, "IMAGE"],
        ],
    },
    "Image2Image": {
        "nodes": [
            {
                "type": "ComfyUIClipInterrogator",
                "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
                "outputs": [
                    {
                        "name": "prompt",
                        "type": "STRING",
                        "links": [],
                        "slot_index": 0,
                    }
                ],
                "pos": [0, 0],
                "index": 0,
            },
            {
                "type": "ComfyUIClipInterrogator",
                "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
                "outputs": [
                    {
                        "name": "prompt",
                        "type": "STRING",
                        "links": [],
                        "slot_index": 0,
                    }
                ],
                "pos": [0, 0],
                "index": 1,
            },
            {
                "type": "EmptyLatentImage",
                "outputs": [
                    {
                        "name": "LATENT",
                        "type": "LATENT",
                        "links": [],
                        "slot_index": 0,
                    }
                ],
                "pos": [0, 0],
                "index": 2,
            },
            {
                "type": "CLIPTextEncode",
                "inputs": [
                    {"name": "clip", "type": "CLIP", "link": None},
                    {
                        "name": "text",
                        "type": "STRING",
                        "link": None,
                        "widget": {"name": "text"},
                    },
                ],
                "pos": [0, 0],
                "outputs": [
                    {
                        "name": "CONDITIONING",
                        "type": "CONDITIONING",
                        "links": [],
                        "slot_index": 0,
                    }
                ],
                "pos": [0, 0],
                "index": 3,
            },
            {
                "type": "CLIPTextEncode",
                "inputs": [
                    {"name": "clip", "type": "CLIP", "link": None},
                    {
                        "name": "text",
                        "type": "STRING",
                        "link": None,
                        "widget": {"name": "text"},
                    },
                ],
                "outputs": [
                    {
                        "name": "CONDITIONING",
                        "type": "CONDITIONING",
                        "links": [],
                        "slot_index": 0,
                    }
                ],
                "pos": [0, 0],
                "index": 4,
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
                    {
                        "name": "LATENT",
                        "type": "LATENT",
                        "links": [],
                        "slot_index": 0,
                    }
                ],
                "pos": [0, 0],
                "index": 5,
            },
            {
                "type": "VAEDecode",
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": None},
                    {"name": "vae", "type": "VAE", "link": None},
                ],
                "outputs": [
                    {
                        "name": "IMAGE",
                        "type": "IMAGE",
                        "links": [],
                    }
                ],
                "pos": [0, 0],
                "index": 6,
            },
            # {
            #     "type": "SaveImage",
            #     "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
            #     "pos": [0, 0],
            #     "index": 7,
            # },
        ],
        "links": [
            [0, 0, 3, 1, None, "STRING"],
            [1, 0, 4, 1, None, "STRING"],
            [3, 0, 5, 1, None, "CONDITIONING"],
            [4, 0, 5, 2, None, "CONDITIONING"],
            [2, 0, 5, 3, None, "LATENT"],
            [5, 0, 6, 0, None, "LATENT"],
            # [6, 0, 7, 0, None, "IMAGE"],
        ],
    },
}
