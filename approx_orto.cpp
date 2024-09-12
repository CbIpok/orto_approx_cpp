#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <future>
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
// Функция для проверки линейной независимости векторов
bool areVectorsLinearlyIndependent(const std::vector<std::vector<double>>& vectors) {
    if (vectors.empty() || vectors[0].empty()) {
        return false;
    }

    // Определяем размерность векторов
    size_t rows = vectors.size();
    size_t cols = vectors[0].size();

    // Создаем матрицу из векторов
    Eigen::MatrixXd matrix(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = vectors[i][j];
        }
    }

    // Вычисляем ранг матрицы
    Eigen::FullPivLU<Eigen::MatrixXd> lu(matrix);
    int rank = lu.rank();

    // Если ранг матрицы равен числу векторов, они линейно независимы
    return rank == std::min(rows, cols);
}
// Функция для аппроксимации вектора в исходном базисе, возвращающая только коэффициенты
std::vector<double> approximate_with_non_orthogonal_basis_orto_std(
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

std::vector<std::vector<double>> process_range(
    const std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis,
    size_t start, size_t end) {

    std::vector<std::vector<double>> local_coofs;
    std::vector<double> mariogramm_t(start);
    std::vector<std::vector<double>> fk_t(basis.size());
    std::copy_n(vector.begin(), start, mariogramm_t.begin());
    for (size_t i = 0; i < basis.size(); i++) {
        fk_t[i].resize(start);
        std::copy_n(basis[i].begin(), start, fk_t[i].begin());
    }

    for (size_t t = start; t < end; ++t) {
        mariogramm_t.push_back(vector[t]);
        for (size_t i = 0; i < basis.size(); i++) {
            fk_t[i].push_back(basis[i][t]);
        }
        local_coofs.push_back(approximate_with_non_orthogonal_basis_orto_std(mariogramm_t, fk_t));
    }

    return local_coofs;
}

std::vector<std::vector<double>> approximate_with_non_orthogonal_basis_orto_std_t(
    const std::vector<double>& vector, const std::vector<std::vector<double>>& basis
)
{
    const size_t num_tasks = 64; // Количество задач
    size_t chunk_size = vector.size() / num_tasks; // Размер блока для каждой задачи

    std::vector<std::future<std::vector<std::vector<double>>>> futures;
    std::vector<std::vector<double>> coofs; // Результирующий вектор
    std::mutex coofs_mutex; // Мьютекс для синхронизации доступа к результату

    // Запускаем 64 задачи
    for (size_t task = 0; task < num_tasks; ++task) {
        size_t start = task * chunk_size;
        size_t end = (task == num_tasks - 1) ? vector.size() : (task + 1) * chunk_size;

        // Создаём асинхронную задачу
        futures.push_back(std::async(std::launch::async, [start, end, &vector, &basis]() {
            return process_range(vector, basis, start, end);
            }));
    }

    // Ожидание результатов от всех задач и объединение результатов
    for (auto& future : futures) {
        try {
            auto result = future.get();
            // Объединение результатов (защищено мьютексом)
            std::lock_guard<std::mutex> guard(coofs_mutex);
            coofs.insert(coofs.end(), result.begin(), result.end());
        }
        catch (const std::exception& e) {
            std::cerr << "Ошибка при выполнении задачи: " << e.what() << std::endl;
        }
    }

    return coofs;
}