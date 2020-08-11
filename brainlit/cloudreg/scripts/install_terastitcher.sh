sudo apt update
sudo apt install -y build-essential cmake libblacs-mpi-dev openmpi-bin
git clone https://github.com/abria/TeraStitcher
mkdir build-terastitcher
cd build-terastitcher
cmake -DWITH_UTILITY_MODULE_teraconverter=ON -DWITH_UTILITY_MODULE_mdatagenerator=ON -DWITH_UTILITY_MODULE_volumeconverter=ON -DWITH_UTILITY_MODULE_mergedisplacements=ON ../TeraStitcher/src
make -j `nproc`
# need ownership of /usr/local to install without sudo
sudo chown -R ubuntu:ubuntu /usr/local/
make install
