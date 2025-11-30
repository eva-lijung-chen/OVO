#echo "Configuring and building Thirdparty/DBoW2 ..."
cd Thirdparty
#cd DBoW2
#mkdir build
#cd build
#cmake .. -DCMAKE_BUILD_TYPE=Release
#make -j
#cd ../..

echo "Configuring and building Thirdparty/g2o ..."

cd g2o
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="../install" \
         -DCMAKE_BUILD_TYPE="Release" 
make -j1 install
cd ../../


#echo "Configuring and building Thirdparty/Sophus ..."
#cd Sophus
#mkdir build
#cd build
#cmake .. -DCMAKE_BUILD_TYPE=Release
#make -j
#cd ../../

cd ../
echo "Uncompress vocabulary ..."

if [ ! -f Vocabulary/ORBvoc.txt ]; then
    cd Vocabulary
    tar -xf ORBvoc.txt.tar.gz
    cd ..
fi

echo "Configuring and building ORB_SLAM3 ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DG2O_DIR=$(pwd)/../Thirdparty/g2o/install/lib/cmake/g2o
make -j1 install
