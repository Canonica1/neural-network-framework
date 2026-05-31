# neural-network-framework

Header-only учебный фреймворк для простых полносвязных нейросетей.

## Зависимости

- C++20 compiler
- CMake 3.16+
- Eigen3

Если Eigen3 не установлен в системе, можно положить исходники Eigen в `third_party/eigen`.

## MNIST

```bash
mkdir -p data/mnist
curl -fL https://raw.githubusercontent.com/phoebetronic/mnist/main/mnist_train.csv.zip \
    -o data/mnist/mnist_train.csv.zip
unzip -o data/mnist/mnist_train.csv.zip -d data/mnist
```

## Сборка и запуск

```bash
cmake -S . -B build
cmake --build build
./build/demo
```

На локальном запуске demo обучается 10 эпох и поднимает validation accuracy выше `0.95`.
