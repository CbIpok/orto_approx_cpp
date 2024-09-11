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

std::vector<Vector> gram_schmidt_with_fixed_first_vector(const std::vector<Vector>& vectors) {
    size_t num_vectors = vectors.size();
    size_t vector_size = vectors[0].size();

    std::vector<Vector> orthogonal_basis;
    orthogonal_basis.reserve(num_vectors);  // ����������� ����� ��� ���� ��������

    // ������ ������ ������� ����������
    orthogonal_basis.push_back(vectors[0]);

    // ���������� ��������� ������ ��� ��������
    for (size_t i = 1; i < num_vectors; ++i) {
        Vector v = vectors[i];

        for (size_t j = 0; j < orthogonal_basis.size(); ++j) {
            const Vector& u = orthogonal_basis[j];

            // ���������������� ������ ����-������: ��������� ��������
            double projection_scale = v.dot(u) / u.squaredNorm();
            v -= projection_scale * u;
        }

        orthogonal_basis.push_back(v);
    }

    return orthogonal_basis;
}

// ������� ��� ���������� ������������� ���������� � �������� ������ � �������������� Eigen
std::vector<double> find_coefficients_in_original_basis(
    const std::vector<Vector>& basis,
    const std::vector<Vector>& orthogonal_basis,
    const Vector& f_bort
) {
    size_t n = basis.size();

    // �������� ������� �������� � �������������� Eigen
    Matrix transition_matrix(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            transition_matrix(i, j) = orthogonal_basis[j].dot(basis[i]);
        }
    }

    // ������������ f_bort �� ������ ������������� ��������
    Vector f_bort_scaled = f_bort;
    for (size_t i = 0; i < n; ++i) {
        f_bort_scaled(i) *= orthogonal_basis[i].dot(orthogonal_basis[i]);
    }

    // ������� ������� ��������� � �������������� ������ QR-����������
    Vector f_b = transition_matrix.transpose().colPivHouseholderQr().solve(f_bort_scaled);

    // �������������� ���������� � std::vector
    return std::vector<double>(f_b.data(), f_b.data() + f_b.size());
}

// ������� ��� ���������� ������� �� �������������� ������
Vector decompose_vector(const Vector& vector, const std::vector<Vector>& basis) {
    Vector coefficients(basis.size());
    for (size_t i = 0; i < basis.size(); ++i) {
        coefficients(i) = vector.dot(basis[i]) / basis[i].dot(basis[i]);
    }
    return coefficients;
}

// ������� ��� ������������� ������� � �������� ������, ������������ ������ ������������
std::vector<double> approximate_with_non_orthogonal_basis_orto(
    const std::vector<double>& vector, const std::vector<std::vector<double>>& basis
) {
    // ����������� ������� ������ � ���������� Eigen
    Vector eigen_vector = to_eigen_vector(vector);
    std::vector<Vector> eigen_basis = to_eigen_basis(basis);

    // �������������� �����
    std::vector<Vector> orthogonal_basis = gram_schmidt_with_fixed_first_vector(eigen_basis);

    // ��������� ������ �� �������������� ������
    Vector f_bort = decompose_vector(eigen_vector, orthogonal_basis);

    // ������� ������������ � �������� ������
    std::vector<double> coefs;
    try {
        coefs = find_coefficients_in_original_basis(eigen_basis, orthogonal_basis, f_bort);
    }
    catch (const std::runtime_error& e) {
        return {};
    }
    return coefs;
}