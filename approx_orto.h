#pragma once
#include <vector>
#include <Eigen/Dense>
// Тип для удобства
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

std::vector<std::vector<double>> approximate_with_non_orthogonal_basis_orto_t(
    const std::vector<double>& vector, const std::vector<std::vector<double>>& basis
);

Vector approximate_with_non_orthogonal_basis_orto(const Vector& x, const Matrix& f_k);

std::vector<double> approximate_with_non_orthogonal_basis_orto_std(
    const std::vector<double>& vector, const std::vector<std::vector<double>>& basis);