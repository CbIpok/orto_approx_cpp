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

// Оптимизированный Модифицированный Грам-Шмидт с фиксацией первого вектора для времени
std::vector<Vector> gram_schmidt_with_fixed_first_vector_t(const std::vector<Vector>& vectors, size_t time_step) {
    std::vector<Vector> orthogonal_basis;
    orthogonal_basis.reserve(vectors.size());

    // Ортогонализируем только до текущего временного шага
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

// Функция для нахождения коэффициентов разложения в исходном базисе с учётом времени
std::vector<double> find_coefficients_in_original_basis_t(
    const std::vector<Vector>& basis,
    const std::vector<Vector>& orthogonal_basis,
    const Vector& f_bort,
    size_t time_step
) {
    size_t n = time_step;

    // Создание матрицы перехода с использованием Eigen
    Matrix transition_matrix(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            transition_matrix(i, j) = orthogonal_basis[j].dot(basis[i]);
        }
    }

    // Нормирование f_bort по нормам ортогональных векторов
    Vector f_bort_scaled = f_bort.head(n);
    for (size_t i = 0; i < n; ++i) {
        f_bort_scaled(i) *= orthogonal_basis[i].dot(orthogonal_basis[i]);
    }

    // Решение системы уравнений с использованием метода QR-разложения
    Vector f_b = transition_matrix.transpose().colPivHouseholderQr().solve(f_bort_scaled);

    // Преобразование результата в std::vector
    return std::vector<double>(f_b.data(), f_b.data() + f_b.size());
}

// Функция для поэтапного разложения вектора с учётом времени
Vector decompose_vector_t(const Vector& vector, const std::vector<Vector>& basis, size_t time_step) {
    Vector coefficients(time_step);
    for (size_t i = 0; i < time_step; ++i) {
        coefficients(i) = vector.head(time_step).dot(basis[i]) / basis[i].dot(basis[i]);
    }
    return coefficients;
}

// Оптимизированная функция для получения коэффициентов аппроксимации на каждом временном шаге
std::vector<std::vector<double>> approximate_with_non_orthogonal_basis_orto_t(
    const std::vector<double>& vector, const std::vector<std::vector<double>>& basis
) {
   
    size_t max_time = vector.size();  // Количество временных шагов
    std::vector<std::vector<double>> all_coefficients(max_time);
  
    // Преобразуем входные данные в контейнеры Eigen
    Vector eigen_vector = to_eigen_vector(vector);
    std::vector<Vector> eigen_basis = to_eigen_basis(basis);
  
    // Цикл по временным шагам
    for (size_t t = 1; t <= max_time; ++t) {
        // Ортогонализуем базис до текущего временного шага
        std::vector<Vector> orthogonal_basis = gram_schmidt_with_fixed_first_vector_t(eigen_basis, t);

        // Разлагаем вектор по ортогональному базису до текущего временного шага
        Vector f_bort = decompose_vector_t(eigen_vector, orthogonal_basis, t);

        // Находим коэффициенты для текущего временного шага
        std::vector<double> coefs = find_coefficients_in_original_basis_t(eigen_basis, orthogonal_basis, f_bort, t);

        // Сохраняем коэффициенты
        all_coefficients[t - 1] = std::move(coefs);
    }

    return all_coefficients;
}
