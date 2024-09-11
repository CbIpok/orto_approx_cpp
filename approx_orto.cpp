﻿#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "approx_orto.h"
// Типы для удобства


// Функция для скалярного произведения двух векторов с использованием Eigen
inline double dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size for dot product.");
    }
    return v1.dot(v2);  // Оптимизированное скалярное произведение из Eigen
}

// Функция для ортогонализации системы векторов по методу Грама-Шмидта с использованием Eigen
Matrix gram_schmidt(const Matrix& vectors) {
    size_t n = vectors.rows();
    size_t m = vectors.cols();

    if (n == 0 || m == 0) {
        throw std::invalid_argument("Matrix must have at least one row and one column.");
    }

    Matrix orthogonal_vectors(n, m);
    orthogonal_vectors.setZero();

    // Вектор для временного хранения промежуточных результатов
    Vector new_vector(m);

    for (size_t i = 0; i < n; ++i) {
        new_vector = vectors.row(i);  // Избегаем копирования, используем прямой доступ к строкам

        if (i == 0) {
            orthogonal_vectors.row(0) = new_vector;
        }
        else {
            for (size_t j = 0; j < i; ++j) {
                // Оптимизация: используем Eigen для эффективного вычисления скалярных произведений и предотвращаем создание временных объектов
                double denom = orthogonal_vectors.row(j).squaredNorm();  // Векторная норма вместо явного dot_product
                if (denom == 0) {
                    throw std::invalid_argument("Zero vector encountered during orthogonalization.");
                }

                double scale = new_vector.dot(orthogonal_vectors.row(j)) / denom;

                // Используем noalias() для предотвращения временных объектов
                new_vector.noalias() -= scale * orthogonal_vectors.row(j);
            }
            orthogonal_vectors.row(i) = new_vector;
        }
    }

    return orthogonal_vectors;
}

Matrix compute_F_matrix(const Matrix& l) {
    size_t n = l.rows();
    Matrix F_matrix = Matrix::Zero(n, n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sum_part = 0;
            for (size_t k = i + 1; k < j; ++k) {
                sum_part += l(j, k) * F_matrix(k, i);
            }
            F_matrix(j, i) = l(j, i) + sum_part;
        }
    }

    

    return F_matrix;
}
// Вычисление bi с проверкой на размеры
Vector compute_bi(size_t k, const Vector& a_k, const Matrix& l) {
    if (k >= a_k.size()) {
        throw std::invalid_argument("Index k is out of bounds for vector a_k.");
    }

    Vector b = Vector::Zero(a_k.size());
    Matrix F_matrix = compute_F_matrix(l);

    if (k >= 1) {
        b[k] = a_k[k];
        b[k - 1] = a_k[k - 1] + a_k[k] * F_matrix(k, k - 1);
    }

    if (k >= 2) {
        b[k - 2] = a_k[k - 2] + a_k[k - 1] * F_matrix(k - 1, k - 2) + a_k[k] * F_matrix(k, k - 2);
    }

    for (int i = static_cast<int>(k) - 3; i >= 0; --i) {
        double sum_part = 0;
        for (size_t j = i + 1; j <= k; ++j) {
            sum_part += a_k[j] * F_matrix(j, i);
        }
        b[i] = a_k[i] + sum_part;
    }

    return b;
}

Vector decompose_vector(const Vector& v, const Matrix& orthogonal_basis) {
    Vector coefficients(orthogonal_basis.rows());

    for (size_t i = 0; i < orthogonal_basis.rows(); ++i) {
        double denominator = dot_product(orthogonal_basis.row(i), orthogonal_basis.row(i));
        coefficients[i] = (denominator != 0) ? dot_product(v, orthogonal_basis.row(i)) / denominator : 0;

        // Îòëàäî÷íàÿ ïå÷àòü
        //std::cout << "Êîýôôèöèåíò " << i << ": " << coefficients[i] << std::endl;
    }

    return coefficients;
}

inline double compute_l_k_i(const Vector& f_k_i, const Vector& e_i) {
    double dot_product_fk_ei = dot_product(f_k_i, e_i);
    double dot_product_ei_ei = dot_product(e_i, e_i);

    return (dot_product_ei_ei == 0) ? 0 : -dot_product_fk_ei / dot_product_ei_ei;
}

// Основная функция для аппроксимации вектора x в базисе f_k с обработкой малых данных
Vector approximate_with_non_orthogonal_basis_orto(const Vector& x, const Matrix& f_k) {
    if (x.size() == 0 || f_k.rows() == 0 || f_k.cols() == 0) {
        throw std::invalid_argument("Input vector and basis matrix must not be empty.");
    }

    // Ортогонализация базиса
    Matrix e_i = gram_schmidt(f_k);

    // Разложение вектора x по ортогональному базису
    Vector a_k = decompose_vector(x, e_i);

    // Вычисление l_k_i
    Matrix l_k_i(f_k.rows(), f_k.cols());

    for (size_t k = 0; k < f_k.rows(); ++k) {
        for (size_t i = 0; i < e_i.rows(); ++i) {
            l_k_i(k, i) = compute_l_k_i(f_k.row(k), e_i.row(i));
        }
    }

    // Вычисление b_i
    size_t k = a_k.size() - 1;
    if (k < 2) {
        throw std::invalid_argument("Basis is too small for the decomposition.");
    }

    Vector b = compute_bi(k, a_k, l_k_i);

    return b;
}



// Функция-обертка для std::vector
std::vector<double> approximate_with_non_orthogonal_basis_orto_std(
    const std::vector<double>& vector, const std::vector<std::vector<double>>& basis) {

    // Преобразуем std::vector<double> в Eigen::VectorXd
    Vector x = Eigen::Map<const Vector>(vector.data(), vector.size());

    // Преобразуем std::vector<std::vector<double>> в Eigen::MatrixXd
    size_t rows = basis.size();
    size_t cols = basis[0].size();
    Matrix f_k(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            f_k(i, j) = basis[i][j];
        }
    }
    Vector result;
    try {
        result = approximate_with_non_orthogonal_basis_orto(x, f_k);
    }
    catch (std::invalid_argument e){
        return {};
    }
    // Вызываем оригинальную функцию с типами Eigen
    

    // Преобразуем результат обратно в std::vector<double>
    std::vector<double> result_std(result.data(), result.data() + result.size());

    return result_std;
}