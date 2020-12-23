sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update -qq
#sudo apt-get install -y cmake gcc gfortran g++ liblapack-dev libopenmpi-dev openmpi-bin
sudo apt-get install -y cmake gcc gfortran g++
sudo apt-get install -y netcdf-bin libnetcdf-dev

# Install and identify the location of the cmocka testing library.
# (The values of the environment variables below don't matter, because
#  cmocka files are installed in standard locations.)
sudo apt-get install -y libcmocka-dev
export CMOCKA_INCLUDE_DIR=/usr/local/include
export CMOCKA_INCLUDE_DIR=/usr/local/lib

