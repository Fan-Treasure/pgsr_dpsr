import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
scenes = [110]
data_base_path='workdir/DTU'
out_base_path='output_dtu'
eval_path='workdir/DTU'
out_name='test'
gpu_id=6

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet -r2 --ncc_scale 0.5"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)
    # CUDA_VISIBLE_DEVICES=1 python train.py -s workdir/DTU/scan24/ -m outputs/debug24/ --quiet -r2 --ncc_scale 0.5

    common_args = "--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)
    # CUDA_VISIBLE_DEVICES=0 python render.py -m outputs/debug24/ --quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}"
    print(cmd)
    os.system(cmd)
    # CUDA_VISIBLE_DEVICES=0 python scripts/eval_dtu/evaluate_single_scene.py --input_mesh outputs/debug24/mesh/tsdf_fusion_post.ply --scan_id 24 --output_dir outputs/debug24/mesh --mask_dir workdir/DTU/ --DTU workdir/DTU/