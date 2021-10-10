# This pulls in the same external libraries and makes them available to Rcpp's
# compilation. We don't use RcppEigen because the version there is too old to
# contain the features used by irlba.

set -e
set -u

rm -f Eigen
ln -s ../../build/_deps/eigen-src/Eigen .

rm -f aarand
ln -s ../../build/_deps/aarand-src/include/aarand .

rm -f irlba
ln -s ../../include/irlba .
