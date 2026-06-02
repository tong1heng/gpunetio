#!/bin/bash

# 0. (re)install mlnx-ofed

/usr/sbin/ofed_uninstall.sh
./mlnxofedinstall --without-dkms --add-kernel-support --without-fw-update --force
/etc/init.d/openibd restart
sudo service opensm restart

# 1. install open-flavor kernel modules

# remove nvidia modules
lsmod |grep ^nvidia |awk '{print $1}'
lsmod |grep ^nvidia |awk '{print $1}' |xargs rmmod

# check if nvidia modules are still in use
lsof /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia0 /dev/nvidiactl
lsof /dev/nvidia*
systemctl status nvidia-persistenced

# install open-gpu-kernel-modules
cd ./third_party/open-gpu-kernel-modules
make clean
make -j
cd kernel-open
sudo insmod ./nvidia.ko NVreg_RegistryDwords="PeerMappingOverride=1" NVreg_EnableStreamMemOPs=1
sudo insmod ./nvidia-modeset.ko
sudo insmod ./nvidia-drm.ko
sudo insmod ./nvidia-uvm.ko
sudo insmod ./nvidia-peermem.ko


# 2. install gdrdrv
cd ./third_party/gdrcopy
# export NVIDIA_SRC_DIR=/usr/src/nvidia-580.95.05/nvidia
export NVIDIA_SRC_DIR=/home/tyh/gpunetio/third_party/open-gpu-kernel-modules/kernel-open/nvidia
make clean
make -j
./insmod.sh


# 3. build gpunetio examples
bear -- make -j8
