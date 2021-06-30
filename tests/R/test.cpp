#include "Rcpp.h"
#include "irlba/irlba.hpp"
#include <algorithm>
#include "Eigen/Dense"

// [[Rcpp::plugins(cpp17)]]

// [[Rcpp::export(rng=false)]]
Rcpp::List run_irlba(Rcpp::NumericMatrix x, Rcpp::NumericVector init, int number, int work) {
    irlba::Irlba irl;
    irlba::NormalSampler norm(50);
    irl.set_number(number).set_work(work);

    Eigen::VectorXd v(init.size());
    std::copy(init.begin(), init.end(), v.data());
    irl.set_init(v);

    // Can't be bothered making a Map, just copying everything for testing purposes.
    Eigen::MatrixXd A(x.nrow(), x.ncol());
    std::copy(x.begin(), x.end(), A.data());

    Eigen::MatrixXd U, V;
    Eigen::VectorXd S;
    irl.run(A, false, false, norm, U, V, S);

    Rcpp::NumericMatrix outU(U.rows(), U.cols());
    std::copy(U.data(), U.data() + U.size(), outU.begin());
    Rcpp::NumericMatrix outV(V.rows(), V.cols());
    std::copy(V.data(), V.data() + V.size(), outV.begin());
    Rcpp::NumericVector outS(S.size());
    std::copy(S.begin(), S.end(), outS.begin());
    
    return Rcpp::List::create(Rcpp::Named("U")=outU, Rcpp::Named("V")=outV, Rcpp::Named("S")=outS);
}
