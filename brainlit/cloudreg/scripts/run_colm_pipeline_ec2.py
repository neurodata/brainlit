from .util import start_ec2_instance, run_command_on_server
import argparse
import boto3

def run_colm_pipeline(
    ssh_key_path,
    instance_id,
    input_s3_path,
    output_s3_path,
    num_channels,
    autofluorescence_channel,
    log_s3_path=None,
    instance_type="r5d.24xlarge",
):
    """Run COLM pipeline on EC2 instance

    Args:
        ssh_key_path (str): Local path to ssh key needed for this server
        instance_id (str): ID of the EC2 instance to run pipeline on
        input_s3_path (str): S3 Path to raw data
        output_s3_path (str): S3 path to store precomputed volume. Volume is stored at output_s3_path/channel for each channel.
        num_channels (int): Number of channels in this volume
        autofluorescence_channel (int): Autofluorescence channel number
        log_s3_path (str, optional): S3 path to store intermediates including vignetting correction and Terastitcher files. Defaults to None.
        instance_type (str, optional): AWS EC2 instance type. Defaults to "r5d.24xlarge".
    """
    # get ec2 client
    ec2 = boto3.resource("ec2")

    public_ip_address = start_ec2_instance(instance_id, instance_type)

    # now run command on instance
    # update the code on the instance
    update_command = "mkdir -p ~/ssd1 ~/ssd2; git clone https://github.com/neurodata/CloudReg.git; cd CloudReg; git pull; docker pull neurodata/cloudreg;"
    print("updating CloudReg code on EC2 instance...")
    errors_update = run_command_on_server(
        update_command, ssh_key_path, public_ip_address
    )
    # mount ssds command
    command1 = "sudo bash CloudReg/scripts/mount_combined_ssds.sh"
    # colm pipeline command
    # command2 = f'time /home/ubuntu/colm_pipeline_env/bin/python CloudReg/scripts/colm_pipeline.py {input_s3_path} {output_s3_path} {num_channels} {autofluorescence_channel} --log_s3_path {log_s3_path}'
    command2 = f"cd CloudReg/; time docker-compose run -v ~/ssd1:/root/ssd1 -v ~/ssd2:/root/ssd2 cloudreg {input_s3_path} {output_s3_path} {num_channels} {autofluorescence_channel} --log_s3_path {log_s3_path}"
    print(command2)
    errors1 = run_command_on_server(command1, ssh_key_path, public_ip_address)
    print(errors1)
    errors2 = run_command_on_server(command2, ssh_key_path, public_ip_address)
    print(errors2)

    # shut down instance
    ec2.meta.client.stop_instances(InstanceIds=[instance_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run COLM pipeline on remote EC2 instance with given input parameters"
    )
    parser.add_argument(
        "-ssh_key_path", help="path to identity file used to ssh into given instance"
    )
    parser.add_argument(
        "-instance_id", help="EC2 Instance ID of instance to run COLM pipeline on."
    )
    parser.add_argument(
        "-input_s3_path",
        help="S3 path to input colm data. Should be of the form s3://<bucket>/<experiment>",
        type=str,
    )
    parser.add_argument(
        "-output_s3_path",
        help="S3 path to store precomputed volume. Precomputed volumes for each channel will be stored under this path. Should be of the form s3://<bucket>/<path_to_precomputed>. The data will be saved at s3://<bucket>/<path_to_precomputed>/CHN0<channel>",
        type=str,
    )
    # parser.add_argument('channel_of_interest', help='Channel of interest in experiment',  type=int)
    parser.add_argument(
        "-num_channels", help="Number of channels in experiment", type=int
    )
    parser.add_argument(
        "-autofluorescence_channel", help="Autofluorescence channel number.", type=int
    )
    parser.add_argument(
        "-log_s3_path",
        help="S3 path at which pipeline intermediates can be stored including bias correctin tile.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--instance_type",
        help="EC2 instance type to run pipeline on. minimum r5d.16xlarge",
        type=str,
        default="r5d.16xlarge",
    )

    args = parser.parse_args()

    run_colm_pipeline(
        args.ssh_key_path,
        args.instance_id,
        args.input_s3_path,
        args.output_s3_path,
        args.num_channels,
        args.autofluorescence_channel,
        args.log_s3_path,
        instance_type=args.instance_type,
    )
