#include "Rcpp.h"
#include "irlba/irlba.hpp"
#include <algorithm>
#include "Eigen/Dense"

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export(rng=false)]]
Rcpp::List run_irlba(Rcpp::NumericMatrix x, Rcpp::NumericVector init, int number, int work) {
    irlba::Irlba irl;
    irl.set_number(number).set_work(work);

    Eigen::VectorXd v(init.size());
    std::copy(init.begin(), init.end(), v.data());

    // Can't be bothered making a Map, just copying everything for testing purposes.
    Eigen::MatrixXd A(x.nrow(), x.ncol());
    std::copy(x.begin(), x.end(), A.data());
    auto res = irl.run(A, irlba::null_rng(), &v);

    Rcpp::NumericMatrix outU(res.U.rows(), res.U.cols());
    std::copy(res.U.data(), res.U.data() + res.U.size(), outU.begin());
    Rcpp::NumericMatrix outV(res.V.rows(), res.V.cols());
    std::copy(res.V.data(), res.V.data() + res.V.size(), outV.begin());
    Rcpp::NumericVector outD(res.D.size());
    std::copy(res.D.begin(), res.D.end(), outD.begin());
    
    return Rcpp::List::create(Rcpp::Named("U")=outU, Rcpp::Named("V")=outV, Rcpp::Named("D")=outD);
}
