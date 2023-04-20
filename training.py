import os
import sys
import subprocess

def main():
    # Download the Docker container
    print("Downloading Docker container...")
    download_command = "docker pull xuejiaqi127/f2dd01d4275e"
    result = subprocess.run(download_command, shell=True, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("Failed to download Docker container. Error:")
        print(result.stderr.decode('utf-8'))
        sys.exit(1)

    print("Docker container downloaded successfully.")

    # Ask the user if they want to run the container
    while True:
        user_input = input("Do you want to run the container? (yes/no): ").lower()
        if user_input == "yes" or user_input == "no":
            break
        else:
            print("Please enter 'yes' or 'no'.")

    # Run the container if the user said yes
    if user_input == "yes":
        # Get user inputs for CUDA_VISIBLE_DEVICES, --bs, and --epoch
        cuda_visible_devices = input("Enter the value for CUDA_VISIBLE_DEVICES (e.g. 0,1,2,3): ")
        bs = input("Enter the value for --bs (e.g. 200): ")
        epoch = input("Enter the value for --epoch (e.g. 1000): ")

        print("Running Docker container...")
        run_command = "sudo docker run -d --gpus all --shm-size 256g xuejiaqi127/f2dd01d4275e /bin/bash -c \"CUDA_VISIBLE_DEVICES={0} python -u /workspace/docker/byol/train.py --exp_id 7 --dataset imagenet --lr 5e-5 --bs {1} --emb 128 --eval_every 5 --method byol --arch resnet18 --epoch {2} --train_file_path '/workspace/docker/data/imagenet100/26_n02106550/train_filelist_0.5.txt' --clf_file_path '/workspace/docker/data/imagenet100/26_n02106550/train_filelist_0.5.txt' --test_file_path '/workspace/docker/data/imagenet100/26_n02106550/test_filelist.txt' --test_t_file_path '/workspace/docker/data/imagenet100/26_n02106550/test_t_filelist.txt' --n_0 2 --n_1 1 --n_2 1 --bs_clf 100 --bs_test 100 --fname '/workspace/docker/byol/checkpoint/imagenet100/clean.pt' --clf_fname '/workspace/docker/byol/logs/train_ep={3}.pt' --target_label 64 --trigger_path '/workspace/docker/poison-generation/triggers/trigger_11.png' --drop 50 20 5 --alpha_1 1 --alpha_2 1 --alpha_3 0 --alpha_4 1\"".format(cuda_visible_devices, bs, epoch, '{}')
        result = subprocess.run(run_command, shell=True, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print("Failed to run Docker container. Error:")
            print(result.stderr.decode('utf-8'))
        else:
            print("Docker container is running in the background.")

if __name__ == "__main__":
    main()
