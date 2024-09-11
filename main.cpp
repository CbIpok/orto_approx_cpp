#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cmath>
#include "approx_orto.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <fstream>
#include "json.hpp"


using json = nlohmann::json;


void show_progress_bar(size_t current, size_t total) {
    int bar_width = 70;
    float progress = (float)current / total;
    int pos = bar_width * progress;

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

// Функция для сравнения векторов
bool vectors_are_close(const std::vector<double>& v1, const std::vector<double>& v2, double tol = 1e-6) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) > tol) return false;
    }
    return true;
}

// Генерация тестов
void run_tests() {
    std::vector<std::pair<std::vector<double>, std::vector<std::vector<double>>>> test_cases = {
    {{4.0, 7.0, 13.0}, {{1.0, 1.0, 0.0}, {0.0, 1.0, 2.0}, {1.0, 0.0, 3.0}}},
    {{5.0, 11.0, 15.0}, {{1.0, 3.0, 5.0}, {2.0, 4.0, 6.0}, {1.0, 2.0, 7.0}}},
    {{2.0, 4.0, 6.0}, {{1.0, 0.0, 2.0}, {0.0, 1.0, 3.0}, {2.0, 1.0, 1.0}}},
    {{6.0, 12.0, 18.0}, {{1.0, 1.0, 1.0}, {0.0, 1.0, 2.0}, {1.0, 0.0, 4.0}}},
    {{7.0, 14.0, 21.0}, {{1.0, 2.0, 3.0}, {2.0, 4.0, 5.0}, {3.0, 5.0, 6.0}}},
    {{9.0, 17.0, 25.0}, {{1.0, 2.0, 0.0}, {2.0, 3.0, 4.0}, {0.0, 1.0, 5.0}}},
    {{5.0, 10.0, 15.0, 20.0}, {{1.0, 2.0, 1.0, 0.0}, {2.0, 1.0, 3.0, 1.0}, {3.0, 1.0, 4.0, 2.0}, {1.0, 3.0, 1.0, 4.0}}},
    {{4.0, 8.0, 12.0, 16.0, 20.0}, {{1.0, 2.0, 0.0, 1.0, 0.0}, {2.0, 3.0, 1.0, 0.0, 1.0}, {3.0, 4.0, 1.0, 2.0, 0.0}, {2.0, 3.0, 0.0, 1.0, 3.0}, {1.0, 0.0, 2.0, 1.0, 4.0}}},
    {{6.0, 9.0, 12.0, 15.0, 18.0, 21.0}, {{1.0, 0.0, 2.0, 1.0, 3.0, 0.0}, {2.0, 1.0, 3.0, 2.0, 1.0, 2.0}, {3.0, 2.0, 1.0, 3.0, 2.0, 1.0}, {0.0, 3.0, 1.0, 2.0, 0.0, 3.0}, {1.0, 2.0, 0.0, 3.0, 2.0, 1.0}, {2.0, 3.0, 2.0, 1.0, 0.0, 2.0}}}
    };

    for (size_t i = 0; i < test_cases.size(); ++i) {
        std::cout << "\nTest " << i + 1 << ":\n";
        auto [vector, basis] = test_cases[i];
        std::vector<double> coefs = approximate_with_non_orthogonal_basis_orto_std(vector, basis);

        // Восстанавливаем аппроксимацию и проверяем её
        std::vector<double> approximation(vector.size(), 0.0);
        for (size_t i = 0; i < basis.size(); ++i) {
            for (size_t j = 0; j < vector.size(); ++j) {
                if (coefs.size() != 0)
                    approximation[j] += coefs[i] * basis[i][j];
            }
        }

        if (!coefs.empty() && vectors_are_close(approximation, vector)) {
            std::cout << "Test " << i + 1 << " passed: coefficients match the original vector.\n";
        }
        else {
            std::cout << "Test " << i + 1 << " failed.\n";
        }
    }
}




// Функция для чтения данных из JSON-файла
void read_json_data(const std::string& file_path, std::vector<double>& mariogramm, std::vector<std::vector<double>>& fk) {
    // Открываем файл
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + file_path);
    }

    // Загружаем JSON
    json data;
    file >> data;

    // Чтение одномерного массива mariogramm
    mariogramm = data["mariogramm"].get<std::vector<double>>();

    // Чтение двумерного массива fk
    fk = data["fk"].get<std::vector<std::vector<double>>>();

    // Закрываем файл
    file.close();
}


void json_read()
{
    // Переменные для хранения данных
    std::vector<double> mariogramm;
    std::vector<std::vector<double>> fk;

    // Путь к файлу с данными
    std::string file_path = "data_6.json";

    try {
        // Чтение данных из JSON-файла
        read_json_data(file_path, mariogramm, fk);
        //std::vector<double> coefs = approximate_with_non_orthogonal_basis_orto_std(mariogramm, fk);
        std::vector<double> mariogramm_t;
        std::vector<std::vector<double>> fk_t(fk.size());
        for (size_t t = 0; t < mariogramm.size(); ++t) {
            mariogramm_t.push_back(mariogramm[t]);
            for (size_t i = 0; i < fk.size(); i++)
            {
                fk_t[i].push_back(fk[i][t]);
            }
            approximate_with_non_orthogonal_basis_orto_std(mariogramm_t, fk_t);
           /* show_progress_bar(t, mariogramm.size());*/
        }
                    
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
    }
}

int main() {
    run_tests();
    json_read();
    return 0;
}