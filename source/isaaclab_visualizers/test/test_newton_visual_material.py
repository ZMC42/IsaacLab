# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer material-writer contracts."""

import io
import json
import struct

import numpy as np
import trimesh
import warp as wp
from isaaclab_physx.renderers.visual_material import FabricVisualMaterialWriter
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer
from isaaclab_visualizers.viser.viser_visualizer import NewtonViewerViser


def test_kit_visualizer_publishes_fabric_visual_material_writer() -> None:
    visualizer = object.__new__(KitVisualizer)
    assert visualizer.visual_material_writer is FabricVisualMaterialWriter


def test_viser_textured_instances_export_dielectric_material() -> None:
    """Textured instances retain their image and do not default to metal in the browser."""
    texture = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    viewer = NewtonViewerViser(port=0, verbose=False)
    try:
        viewer.log_mesh(
            "/mesh",
            wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device="cpu"),
            wp.array([0, 1, 2], dtype=wp.int32, device="cpu"),
            uvs=wp.array([[0, 0], [1, 0], [0, 1]], dtype=wp.vec2, device="cpu"),
            texture=texture,
            hidden=True,
        )
        viewer.log_instances(
            "/textured",
            "/mesh",
            wp.array([[0, 0, 0, 0, 0, 0, 1]], dtype=wp.transform, device="cpu"),
            scales=None,
            colors=None,
            materials=None,
        )
        glb_data = viewer._server.scene.get_handle_by_name("/textured").glb_data
        json_length = struct.unpack_from("<I", glb_data, 12)[0]
        document = json.loads(glb_data[20 : 20 + json_length])
        material = document["materials"][0]["pbrMetallicRoughness"]
        # glTF defaults an omitted metallicFactor to 1, even for a plain color texture.
        assert material.get("metallicFactor", 1.0) == 0.0
        mesh = trimesh.load(io.BytesIO(glb_data), file_type="glb", force="mesh", process=False)
        np.testing.assert_array_equal(np.asarray(mesh.visual.material.baseColorTexture), texture)
    finally:
        viewer.close()
