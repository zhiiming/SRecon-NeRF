import open3d as o3d
import json
import numpy as np
import math


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               # top-left image corner
                               [-half_w, -half_h, frustum_length],
                               # top-right image corner
                               [half_w, -half_h, frustum_length],
                               # bottom-right image corner
                               [half_w, half_h, frustum_length],
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(
        1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape(
        (1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(
        np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((0, 0.7, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=4., origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    idx = 0
    for color, full_dict in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []
        fx = full_dict["fl_x"]
        fy = full_dict["fl_y"]
        cx = full_dict["cx"]
        cy = full_dict["cy"]
        K = [fx, 0, cx, 0,
             0, fy, cy, 0,
             0, 0, 1, 0,
             0, 0, 0, 1]
        K = np.array(K).reshape(4, 4)
        camera_dict = full_dict["frames"]

        for cnt, img_name_file in enumerate(camera_dict):
            if cnt % 2 != 0:
                continue
            img_name = img_name_file["file_path"]
            W2C = np.array(img_name_file['transform_matrix']).reshape((4, 4))
            cvToGl = np.zeros((4, 4))
            cvToGl[0, 0] = 1
            cvToGl[1, 1] = -1
            cvToGl[2, 2] = -1
            cvToGl[3, 3] = 1
            W2C = np.dot(cvToGl, W2C)
            C2W = np.linalg.inv(W2C)
            img_size = [full_dict["w"], full_dict["h"]]
            frustums.append(get_camera_frustum(
                img_size, K, C2W, frustum_length=camera_size, color=color))
            cnt += 1
            # if cnt > 20:
            #     break

        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)


def phrase_semantic_nerf_txt():
    txt_path = '/home/ilab/dzm/GitHub/extrinsic2pyramid/data_for_show/traj_w_c.txt'
    Ts_full = np.loadtxt(txt_path, delimiter=" ")  # (900, 16)
    # Ts_full = Ts_full.reshape(-1, 4, 4)
    H = 480
    W = 640
    hfov = 90
    cx = (W - 1.0) / 2.0  # 320
    cy = (H - 1.0) / 2.0  # 240
    fx = W / 2.0 / math.tan(math.radians(hfov / 2.0))  # 320
    fy = fx  # 320
    camera_dict = {}
    for idx in range(900):
        file_name = 'rgb_%d.png' % idx
        camera_dict[file_name] = {}
        camera_dict[file_name]["K"] = [fx, 0, cx, 0,
                                       0, fy, cy, 0,
                                       0, 0, 1, 0,
                                       0, 0, 0, 1]
        T_wc_cv = Ts_full[idx].tolist()
        T_wc_np = Ts_full[idx].reshape(4, 4)
        cvToGl = np.zeros((4, 4))
        cvToGl[0, 0] = 1
        cvToGl[1, 1] = -1
        cvToGl[2, 2] = -1
        cvToGl[3, 3] = 1
        gl_M = np.dot(cvToGl, T_wc_np)
        T_wc_gl = gl_M.flatten().tolist()
        # camera_dict[file_name]["W2C"] = T_wc_gl
        camera_dict[file_name]["W2C"] = T_wc_cv
        camera_dict[file_name]["img_size"] = [640, 480]
    return camera_dict


def write_nerfpp_to_txt():
    matrix_list = []
    cam_dir = '/home/ilab/dzm/GitHub/NeuS/public_data/Replica_room_0_nerfpp/nerfpp_output/posed_images/kai_cameras.json'
    txt_out = '/home/ilab/dzm/GitHub/semantic_nerf/data/Replica/room_0/Sequence_1_my3/traj_w_c.txt'
    camera_dict = json.load(open(cam_dir))
    for idx in range(900):
        img_name = 'rgb_%d.png' % idx
        w2c = camera_dict[img_name]['W2C']
        matrix_list.append(w2c)
    np.savetxt(txt_out, matrix_list, delimiter=" ")


def get_poses():
    cam_dir = '/home/ilab/dzm/GitHub/nerf_for_ipm/dataset/video_kb714/transforms.json'
    camera_dict = json.load(open(cam_dir))
    nn = 1


if __name__ == '__main__':
    import os

    # base_dir = './'
    cam_dir = '/home/ilab/dzm/GitHub/nerf_for_ipm/dataset/video_kb714/transforms.json'
    cam_dir_2 = '/home/ilab/dzm/GitHub/nerf_for_ipm/dataset/video_kb714/transforms_origin.json'
    # write_nerfpp_to_txt()
    # test_cam_dict = phrase_semantic_nerf_txt()

    sphere_radius = 8.
    train_cam_dict = json.load(open(cam_dir))
    test_cam_dict = json.load(open(cam_dir_2))
    # test_cam_dict = json.load(open(os.path.join(base_dir, 'test/cam_dict_norm.json')))
    # path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
    camera_size = 1.0
    # colored_camera_dicts = [([0, 1, 0], train_cam_dict),
    #                         ([0, 0, 1], test_cam_dict),
    #                         ([1, 1, 0], path_cam_dict)
    #                         ]
    colored_camera_dicts = [([1, 0, 0], train_cam_dict),
                            ([0, 0, 1], test_cam_dict),]

    # colored_camera_dicts = [([0, 1, 0], train_cam_dict)]

    geometry_file = '/home/ilab/dzm/GitHub/nerf_for_ipm/exports/pcd/point_cloud.ply'
    geometry_file = None
    geometry_type = 'pointcloud'

    visualize_cameras(colored_camera_dicts, sphere_radius,
                      camera_size=camera_size, geometry_file=geometry_file, geometry_type=geometry_type)
