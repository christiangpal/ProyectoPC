#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>

// --- Configuraciones de la Imagen ---
const int WIDTH = 7680;      // Resolución 8K
const int HEIGHT = 4320;
const int MAX_ITER = 256;    // Iteraciones para Mandelbrot

// --- Kernel para el Filtro de Convolución Gaussiano 5x5 ---
const int KERNEL_SIZE = 5;
const float kernel[5][5] = {
    {1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256},
    {4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256},
    {6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256},
    {4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256},
    {1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256}
};

// ---------------------------------------------------------
// Tarea A: Generación del Conjunto de Mandelbrot
// ---------------------------------------------------------
void generateMandelbrot(std::vector<unsigned char>& image) {
    // Definir los límites del plano complejo
    const float x_min = -2.5f;
    const float x_max = 1.0f;
    const float y_min = -1.0f;
    const float y_max = 1.0f;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            // Mapear el píxel (x, y) al plano complejo (cx, cy)
            float cx = x_min + (x / (float)WIDTH) * (x_max - x_min);
            float cy = y_min + (y / (float)HEIGHT) * (y_max - y_min);

            float zx = 0.0f;
            float zy = 0.0f;
            int iter = 0;

            // Z_{n+1} = Z_n^2 + C
            while (zx * zx + zy * zy < 4.0f && iter < MAX_ITER) {
                float tmp = zx * zx - zy * zy + cx;
                zy = 2.0f * zx * zy + cy;
                zx = tmp;
                iter++;
            }

            // Normalizar a valor de píxel (0-255)
            int pixel_val = (iter == MAX_ITER) ? 0 : (iter * 255 / MAX_ITER);
            image[y * WIDTH + x] = static_cast<unsigned char>(pixel_val);
        }
    }
}

// ---------------------------------------------------------
// Tarea B: Aplicación de Filtro de Convolución 2D
// ---------------------------------------------------------
void applyConvolution(const std::vector<unsigned char>& input, std::vector<unsigned char>& output) {
    int offset = KERNEL_SIZE / 2;

    for (int y = offset; y < HEIGHT - offset; ++y) {
        for (int x = offset; x < WIDTH - offset; ++x) {
            float sum = 0.0f;

            // Aplicar el kernel 5x5
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

// ---------------------------------------------------------
// Utilidad: Guardar Imagen en formato PGM
// ---------------------------------------------------------
void savePGM(const std::string& filename, const std::vector<unsigned char>& image) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error al abrir el archivo para escritura: " << filename << "\n";
        return;
    }
    // Encabezado PGM: Formato P5 (Binario), Ancho, Alto, Valor Máximo de Color
    file << "P5\n" << WIDTH << " " << HEIGHT << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()), image.size());
    file.close();
}

// ---------------------------------------------------------
// Función Principal
// ---------------------------------------------------------
int main() {
    std::cout << "Inicializando memoria para resolucion 8K...\n";
    std::vector<unsigned char> image_original(WIDTH * HEIGHT, 0);
    std::vector<unsigned char> image_filtered(WIDTH * HEIGHT, 0);

    // --- TAREA A ---
    std::cout << "Tarea A: Generando Mandelbrot..." << std::flush;
    auto start_a = std::chrono::high_resolution_clock::now();
    
    generateMandelbrot(image_original);
    
    auto end_a = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_a = end_a - start_a;
    std::cout << " [Completado en " << diff_a.count() << " segundos]\n";

    // Guardar original para comparar
    savePGM("mandelbrot_8k_original.pgm", image_original);

    // --- TAREA B ---
    std::cout << "Tarea B: Aplicando Convolucion (Filtro Gaussiano 5x5)..." << std::flush;
    auto start_b = std::chrono::high_resolution_clock::now();
    
    applyConvolution(image_original, image_filtered);
    
    auto end_b = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_b = end_b - start_b;
    std::cout << " [Completado en " << diff_b.count() << " segundos]\n";

    // Guardar filtrada
    savePGM("mandelbrot_8k_filtered.pgm", image_filtered);

    std::cout << "Proceso completado exitosamente. Imagenes guardadas en el directorio actual.\n";
    return 0;
}