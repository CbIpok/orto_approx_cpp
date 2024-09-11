#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "approx_orto.h"

// ������������� Eigen ��� �������� � ������
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// �������������� std::vector<double> � Eigen::VectorXd
Vector to_eigen_vector(const std::vector<double>& v) {
    return Eigen::Map<const Vector>(v.data(), v.size());
}

// �������������� std::vector<std::vector<double>> � std::vector<Eigen::VectorXd>
std::vector<Vector> to_eigen_basis(const std::vector<std::vector<double>>& basis) {
    std::vector<Vector> eigen_basis;
    eigen_basis.reserve(basis.size());
    for (const auto& vec : basis) {
        eigen_basis.emplace_back(to_eigen_vector(vec));
    }
    return eigen_basis;
}

// ���������������� ���������������� ����-����� � ��������� ������� ������� ��� �������
std::vector<Vector> gram_schmidt_with_fixed_first_vector_t(const std::vector<Vector>& vectors, size_t time_step) {
    std::vector<Vector> orthogonal_basis;
    orthogonal_basis.reserve(vectors.size());

    // ���������������� ������ �� �������� ���������� ����
    for (size_t i = 0; i < time_step; ++i) {
        Vector v = vectors[i];

        for (size_t j = 0; j < orthogonal_basis.size(); ++j) {
            const Vector& u = orthogonal_basis[j];
            double projection_scale = v.dot(u) / u.squaredNorm();
            v -= projection_scale * u;
        }

        orthogonal_basis.push_back(v);
    }

    return orthogonal_basis;
}

// ������� ��� ���������� ������������� ���������� � �������� ������ � ������ �������
std::vector<double> find_coefficients_in_original_basis_t(
    const std::vector<Vector>& basis,
    const std::vector<Vector>& orthogonal_basis,
    const Vector& f_bort,
    size_t time_step
) {
    size_t n = time_step;

    // �������� ������� �������� � �������������� Eigen
    Matrix transition_matrix(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            transition_matrix(i, j) = orthogonal_basis[j].dot(basis[i]);
        }
    }

    // ������������ f_bort �� ������ ������������� ��������
    Vector f_bort_scaled = f_bort.head(n);
    for (size_t i = 0; i < n; ++i) {
        f_bort_scaled(i) *= orthogonal_basis[i].dot(orthogonal_basis[i]);
    }

    // ������� ������� ��������� � �������������� ������ QR-����������
    Vector f_b = transition_matrix.transpose().colPivHouseholderQr().solve(f_bort_scaled);

    // �������������� ���������� � std::vector
    return std::vector<double>(f_b.data(), f_b.data() + f_b.size());
}

// ������� ��� ���������� ���������� ������� � ������ �������
Vector decompose_vector_t(const Vector& vector, const std::vector<Vector>& basis, size_t time_step) {
    Vector coefficients(time_step);
    for (size_t i = 0; i < time_step; ++i) {
        coefficients(i) = vector.head(time_step).dot(basis[i]) / basis[i].dot(basis[i]);
    }
    return coefficients;
}

// ���������������� ������� ��� ��������� ������������� ������������� �� ������ ��������� ����
std::vector<std::vector<double>> approximate_with_non_orthogonal_basis_orto_t(
    const std::vector<double>& vector, const std::vector<std::vector<double>>& basis
) {
   
    size_t max_time = vector.size();  // ���������� ��������� �����
    std::vector<std::vector<double>> all_coefficients(max_time);
  
    // ����������� ������� ������ � ���������� Eigen
    Vector eigen_vector = to_eigen_vector(vector);
    std::vector<Vector> eigen_basis = to_eigen_basis(basis);
  
    // ���� �� ��������� �����
    for (size_t t = 1; t <= max_time; ++t) {
        // �������������� ����� �� �������� ���������� ����
        std::vector<Vector> orthogonal_basis = gram_schmidt_with_fixed_first_vector_t(eigen_basis, t);

        // ��������� ������ �� �������������� ������ �� �������� ���������� ����
        Vector f_bort = decompose_vector_t(eigen_vector, orthogonal_basis, t);

        // ������� ������������ ��� �������� ���������� ����
        std::vector<double> coefs = find_coefficients_in_original_basis_t(eigen_basis, orthogonal_basis, f_bort, t);

        // ��������� ������������
        all_coefficients[t - 1] = std::move(coefs);
    }

    return all_coefficients;
}
