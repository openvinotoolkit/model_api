#!/bin/bash

echo wgetwgetwgetwgetwgetwgetwgetwgetwgetwgetwgetwgetwget
# Added required keys / do the update
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

echo aptkeyaptkeyaptkeyaptkeyaptkeyaptkeyaptkeyaptkeyaptkeyaptkey
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

echo teeteeteeteeteeteeteeteeteeteeteeteeteeteeteetee
echo "deb https://apt.repos.intel.com/openvino/2022 focal main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2022.list

echo updateupdateupdateupdateupdateupdateupdateupdateupdateupdateupdateupdateupdateupdateupdateupdateupdateupdateupdate
apt update

#Install OpenCV
echo libopencvlibopencvlibopencvlibopencvlibopencvlibopencvlibopencvlibopencvlibopencvlibopencv
apt-get install libopencv-dev

# Install OpenVINO
echo openvinoopenvinoopenvinoopenvinoopenvinoopenvinoopenvinoopenvinoopenvinoopenvinoopenvinoopenvinoopenvino
apt install openvino
