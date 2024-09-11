#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "approx_orto.h"


// Использование Eigen для векторов и матриц
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

// Преобразование std::vector<double> в Eigen::VectorXd
Vector to_eigen_vector(const std::vector<double>& v) {
    return Eigen::Map<const Vector>(v.data(), v.size());
}

// Преобразование std::vector<std::vector<double>> в std::vector<Eigen::VectorXd>
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
    orthogonal_basis.reserve(num_vectors);  // Резервируем место для всех векторов

    // Первый вектор остаётся неизменным
    orthogonal_basis.push_back(vectors[0]);

    // Используем временные буферы для проекций
    for (size_t i = 1; i < num_vectors; ++i) {
        Vector v = vectors[i];

        for (size_t j = 0; j < orthogonal_basis.size(); ++j) {
            const Vector& u = orthogonal_basis[j];

            // Модифицированная версия Грам-Шмидта: вычитание проекции
            double projection_scale = v.dot(u) / u.squaredNorm();
            v -= projection_scale * u;
        }

        orthogonal_basis.push_back(v);
    }

    return orthogonal_basis;
}

// Функция для нахождения коэффициентов разложения в исходном базисе с использованием Eigen
std::vector<double> find_coefficients_in_original_basis(
    const std::vector<Vector>& basis,
    const std::vector<Vector>& orthogonal_basis,
    const Vector& f_bort
) {
    size_t n = basis.size();

    // Создание матрицы перехода с использованием Eigen
    Matrix transition_matrix(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            transition_matrix(i, j) = orthogonal_basis[j].dot(basis[i]);
        }
    }

    // Нормирование f_bort по нормам ортогональных векторов
    Vector f_bort_scaled = f_bort;
    for (size_t i = 0; i < n; ++i) {
        f_bort_scaled(i) *= orthogonal_basis[i].dot(orthogonal_basis[i]);
    }

    // Решение системы уравнений с использованием метода QR-разложения
    Vector f_b = transition_matrix.transpose().colPivHouseholderQr().solve(f_bort_scaled);

    // Преобразование результата в std::vector
    return std::vector<double>(f_b.data(), f_b.data() + f_b.size());
}

// Функция для разложения вектора по ортогональному базису
Vector decompose_vector(const Vector& vector, const std::vector<Vector>& basis) {
    Vector coefficients(basis.size());
    for (size_t i = 0; i < basis.size(); ++i) {
        coefficients(i) = vector.dot(basis[i]) / basis[i].dot(basis[i]);
    }
    return coefficients;
}

// Функция для аппроксимации вектора в исходном базисе, возвращающая только коэффициенты
std::vector<double> approximate_with_non_orthogonal_basis_orto(
    const std::vector<double>& vector, const std::vector<std::vector<double>>& basis
) {
    // Преобразуем входные данные в контейнеры Eigen
    Vector eigen_vector = to_eigen_vector(vector);
    std::vector<Vector> eigen_basis = to_eigen_basis(basis);

    // Ортогонализуем базис
    std::vector<Vector> orthogonal_basis = gram_schmidt_with_fixed_first_vector(eigen_basis);

    // Разлагаем вектор по ортогональному базису
    Vector f_bort = decompose_vector(eigen_vector, orthogonal_basis);

    // Находим коэффициенты в исходном базисе
    std::vector<double> coefs;
    try {
        coefs = find_coefficients_in_original_basis(eigen_basis, orthogonal_basis, f_bort);
    }
    catch (const std::runtime_error& e) {
        return {};
    }
    return coefs;
}