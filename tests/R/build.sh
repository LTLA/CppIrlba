# This pulls in the same external libraries and makes them available to Rcpp's
# compilation. We don't use RcppEigen because the version there is too old to
# contain the features used by irlba.

set -e
set -u

rm -f extern
ln -s ../../extern 

cat << EOF > CMakeLists.txt
cmake_minimum_required(VERSION 3.14)

project(irlba-tests)

add_subdirectory(extern)
EOF

cmake -S . -B build
cmake --build build

rm -f Eigen
ln -s build/_deps/eigen-src/Eigen .
