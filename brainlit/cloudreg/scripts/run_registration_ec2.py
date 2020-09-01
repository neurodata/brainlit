# local imports
from .util import start_ec2_instance, run_command_on_server
from .visualization import (
    create_viz_link,
    ara_average_data_link,
)
from .registration import get_affine_matrix

import argparse
import boto3

python_path = "/home/ubuntu/colm_pipeline_env/bin/python"


def run_registration(
    ssh_key_path,
    instance_id,
    instance_type,
    input_s3_path,
    output_s3_path,
    log_s3_path,
    initial_translation,
    initial_rotation,
    orientation,
    fixed_scale,
    missing_data_correction,
    grid_correction,
    bias_correction,
    sigma_regularization,
    num_iterations,
):
    """Run EM-LDDMM registration on an AWS EC2 instance

    Args:
        ssh_key_path (str): Local path to ssh key for this server
        instance_id (str): ID of EC2 instance to use
        instance_type (str): AWS EC2 instance type. Recommended is r5.8xlarge
        input_s3_path (str): S3 path to precomputed data to be registered
        output_s3_path (str): S3 path to store precomputed volume of atlas transformed to input data
        log_s3_path (str): S3 path to store intermediates at
        initial_translation (list of float): Initial translations in x,y,z of input data
        initial_rotation (list): Initial rotation in x,y,z for input data
        orientation (str): 3-letter orientation of input data
        fixed_scale (float): Isotropic scale factor on input data    
        missing_data_correction (bool): Perform missing data correction to ignore zeros in image
        grid_correction (bool): Perform grid correction (for COLM data)
        bias_correction (bool): Perform illumination correction
        sigma_regularization (float): Regularization constat in cost function. Higher regularization constant means less regularization
        num_iterations (int): Number of iterations of EM-LDDMM to run
    """

    # this is the initialization for registration
    atlas_res = 50
    atlas_orientation = "PIR"
    atlas_s3_path = ara_average_data_link(atlas_res)
    atlas_affine_initialization = get_affine_matrix(
        initial_translation,
        initial_rotation,
        atlas_orientation,
        orientation,
        fixed_scale,
        atlas_s3_path,
        center=True,
    )
    target_affine = get_affine_matrix(
        [0] * 3, [0] * 3, orientation, orientation, 1.0, input_s3_path, center=True
    )

    # get viz link from input link
    viz_link = create_viz_link(
        [input_s3_path, atlas_s3_path],
        affine_matrices=[target_affine, atlas_affine_initialization],
    )

    # ask user if this initialization looks right
    user_input = ""
    while user_input == "":
        user_input = input(f"Does this initialization look right? {viz_link} (y/n): ")
    # if no quit and ask for another initialization
    if user_input == "n":
        raise (Exception("Please rerun with new initialization"))
    # else continue
    # start ec2 instance
    public_ip_address = start_ec2_instance(instance_id, instance_type)

    # now run command on instance
    # update the code on the instance
    update_command = "cd ~/CloudReg; git pull;"
    _ = run_command_on_server(update_command, ssh_key_path, public_ip_address)
    # matlab registration command
    command2 = f"time {python_path} CloudReg/scripts/registration.py -input_s3_path {input_s3_path} --output_s3_path {output_s3_path} -orientation {orientation} --rotation {' '.join(map(str,initial_rotation))} --translation {' '.join(map(str,initial_translation))} --scale {fixed_scale} -log_s3_path {log_s3_path} --missing_data_correction {missing_data_correction} --grid_correction {grid_correction} --bias_correction {bias_correction} --regularization {sigma_regularization} --iterations {num_iterations}"
    errors2 = run_command_on_server(command2, ssh_key_path, public_ip_address)
    print(f"errors: {errors2}")

    # shut down instance
    ec2 = boto3.resource("ec2")
    ec2.meta.client.stop_instances(InstanceIds=[instance_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run COLM pipeline on remote EC2 instance with given input parameters"
    )

    # instance params
    parser.add_argument(
        "-ssh_key_path", help="path to identity file used to ssh into given instance"
    )
    parser.add_argument(
        "-instance_id", help="EC2 Instance ID of instance to run COLM pipeline on."
    )
    parser.add_argument(
        "--instance_type",
        help="EC2 instance type to run registration on. Default is r5.8xlarge",
        type=str,
        default="r5.8xlarge",
    )

    # data params
    parser.add_argument(
        "-input_s3_path",
        help="S3 path to precomputed volume used to register the data",
        type=str,
    )
    parser.add_argument(
        "-output_s3_path",
        help="S3 path to store precomputed volume. Precomputed volumes for each channel will be stored under this path. Should be of the form s3://<bucket>/<path_to_precomputed>. The data will be saved at s3://<bucket>/<path_to_precomputed>/CHN0<channel>",
        type=str,
    )
    parser.add_argument(
        "-log_s3_path",
        help="S3 path at which registration outputs are stored.",
        type=str,
    )

    # affine initialization
    parser.add_argument(
        "-orientation", help="3-letter orientation of data. i.e. LPS", type=str
    )
    parser.add_argument(
        "--fixed_scale",
        help="Fixed scale of data, uniform in all dimensions. Default is 1.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--xy",
        help="Rotation in XY plane in degrees. Default is 0.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--xz",
        help="Rotation in XZ plane in degrees. Default is 0.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--yz",
        help="Rotation in YZ plane in degrees. Default is 0.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--x",
        help="Translation in X axis in microns. Default is 0.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--y",
        help="Translation in Y axis in microns. Default is 0.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--z",
        help="Translation in Z axis in microns. Default is 0.",
        type=float,
        default=0,
    )

    # registration preprocessing params
    parser.add_argument(
        "--missing_data_correction",
        help="Perform missing data correction by ignoring 0 values in image prior to registration.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--grid_correction",
        help="Perform correction for low-intensity grid artifact (COLM data)",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--bias_correction",
        help="Perform bias correction prior to registration.",
        type=bool,
        default=False,
    )

    # registration params
    parser.add_argument(
        "--regularization",
        help="Weight of the regularization. Bigger value means less regularization. Default is 10000",
        type=float,
        default=1e4,
    )
    parser.add_argument(
        "--iterations",
        help="Number of iterations to do at low resolution. Default is 5000.",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    run_registration(
        args.ssh_key_path,
        args.instance_id,
        args.instance_type,
        args.input_s3_path,
        args.output_s3_path,
        args.log_s3_path,
        [args.x, args.y, args.z],
        [args.yz, args.xz, args.xy],
        args.orientation,
        args.fixed_scale,
        args.missing_data_correction,
        args.grid_correction,
        args.bias_correction,
        args.regularization,
        args.iterations,
    )
