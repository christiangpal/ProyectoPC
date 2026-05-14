#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <omp.h> // Libreria OpenMP obligatoria

const int WIDTH = 7680;
const int HEIGHT = 4320;
const int MAX_ITER = 256;

const int KERNEL_SIZE = 5;
const float kernel[5][5] = {
    {1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256},
    {4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256},
    {6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256},
    {4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256},
    {1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256}
};

void generateMandelbrot(std::vector<unsigned char>& image) {
    const float x_min = -2.5f;
    const float x_max = 1.0f;
    const float y_min = -1.0f;
    const float y_max = 1.0f;

    // Directiva OpenMP para paralelizar la Tarea A
    //Resultado de los 3 experimentos: schedule(static, 50) = 1.916s, schedule(dynamic, 50) = 1.21s, schedule(guided, 50) = 1.348s
    //#pragma omp parallel for schedule(static)
    //#pragma omp parallel for schedule(guided, 50)

    
    #pragma omp parallel for schedule(dynamic, 50) //Mejor opcion
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            float cx = x_min + (x / (float)WIDTH) * (x_max - x_min);
            float cy = y_min + (y / (float)HEIGHT) * (y_max - y_min);
            float zx = 0.0f;
            float zy = 0.0f;
            int iter = 0;

            while (zx * zx + zy * zy < 4.0f && iter < MAX_ITER) {
                float tmp = zx * zx - zy * zy + cx;
                zy = 2.0f * zx * zy + cy;
                zx = tmp;
                iter++;
            }

            int pixel_val = (iter == MAX_ITER) ? 0 : (iter * 255 / MAX_ITER);
            image[y * WIDTH + x] = static_cast<unsigned char>(pixel_val);
        }
    }
}

void applyConvolution(const std::vector<unsigned char>& input, std::vector<unsigned char>& output) {
    int offset = KERNEL_SIZE / 2;

    // Directiva OpenMP para paralelizar la Tarea B
    #pragma omp parallel for
    for (int y = offset; y < HEIGHT - offset; ++y) {
        for (int x = offset; x < WIDTH - offset; ++x) {
            float sum = 0.0f;
            for (int ky = -offset; ky <= offset; ++ky) {
                for (int kx = -offset; kx <= offset; ++kx) {
                    int pixel_val = input[(y + ky) * WIDTH + (x + kx)];
                    sum += pixel_val * kernel[ky + offset][kx + offset];
                }
            }
            output[y * WIDTH + x] = static_cast<unsigned char>(sum);
        }
    }
}

void savePGM(const std::string& filename, const std::vector<unsigned char>& image) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return;
    file << "P5\n" << WIDTH << " " << HEIGHT << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), image.size());
    file.close();
}

int main() {
    std::cout << "Inicializando memoria para resolucion 8K...\n";
    std::vector<unsigned char> image_original(WIDTH * HEIGHT, 0);
    std::vector<unsigned char> image_filtered(WIDTH * HEIGHT, 0);

    std::cout << "Tarea A: Generando Mandelbrot..." << std::flush;
    auto start_a = omp_get_wtime();
    generateMandelbrot(image_original);
    auto end_a = omp_get_wtime();
    std::cout << " [Completado en " << (end_a - start_a) << " segundos]\n";
    savePGM("mandelbrot_8k_original.pgm", image_original);

    std::cout << "Tarea B: Aplicando Convolucion..." << std::flush;
    auto start_b = omp_get_wtime();
    applyConvolution(image_original, image_filtered);
    auto end_b = omp_get_wtime();
    std::cout << " [Completado en " << (end_b - start_b) << " segundos]\n";
    savePGM("mandelbrot_8k_filtered.pgm", image_filtered);

    std::cout << "Proceso completado exitosamente.\n";
    return 0;
}