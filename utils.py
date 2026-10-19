"""
DO NOT MODIFY.
"""

import base64
import hashlib
import os
import pickle
import zlib
from collections import namedtuple

Obstacle = namedtuple(
    "Obstacle", ["cx", "cy", "width", "height", "angle", "v", "omega"]
)


def _get_key(scene_name: str, length: int = 256) -> bytes:
    h = hashlib.sha256(f"q2_{scene_name}".encode()).digest()
    key = b""
    i = 0
    while len(key) < length:
        key += hashlib.sha256(h + str(i).encode()).digest()
        i += 1
    return key[:length]


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    key_len = len(key)
    return bytes([data[i] ^ key[i % key_len] for i in range(len(data))])


def decrypt(encrypted_str: str, scene_name: str) -> list:
    encrypted = base64.b64decode(encrypted_str.encode("ascii"))
    key = _get_key(scene_name, len(encrypted))
    compressed = _xor_bytes(encrypted, key)
    data = zlib.decompress(compressed)
    return pickle.loads(data)


def load_obstacles(scene: str, data_dir: str = "data") -> tuple:
    filepath = os.path.join(data_dir, f"{scene}.obs")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Obstacle file not found: {filepath}")

    with open(filepath, "r") as f:
        lines = f.readlines()

    scene_name = None
    encrypted_data = None
    for line in lines:
        line = line.strip()
        if line.startswith("SCENE:"):
            scene_name = line[6:]
        elif line.startswith("DATA:"):
            encrypted_data = line[5:]

    if scene_name is None or encrypted_data is None:
        raise ValueError("Invalid obstacle file format")

    obs_list = decrypt(encrypted_data, scene_name)
    return tuple(
        Obstacle(
            cx=o[0], cy=o[1], width=o[2], height=o[3], angle=o[4], v=o[5], omega=o[6]
        )
        for o in obs_list
    )
