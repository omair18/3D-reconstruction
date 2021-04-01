# Сервис 3D-реконструкции. [(EN)](./doc/README(EN).md)

## Содержание

 [1 . Краткое описание проекта](#brief)   
 [2 . Описание требований](#requirements-desciption)  
 [3 . Описание зависимостей](#dependencies-desciption)  
 [4 . Описание модулей проекта](#modules-desciption)  
 [5 . Инструкция по развёртыванию](#deploying)  
 [6 . Конфигурирование](#configuring)  
 [7 . Использование](#usage)  

<a name="brief"></a>

### 1. Краткое описание проекта  

Данный проект представляет из себя сервис по посторению 3D модели объекта по нескольким видам. Входные данные сервис получает посредством Kafka-сообщений, в которых хранятся изображения объекта с разных сторон.

<a name="requirements-desciption"></a>

### 2. Описание требований  

<a name="dependencies-desciption"></a>

### 3. Описание зависимостей  

Данный проект использует следующие зависимости:

* [CUDA 11.2](https://developer.nvidia.com/cuda-downloads)

* [Boost 1.75.0](https://github.com/boostorg/boost/tree/boost-1.75.0)  

* [OpenCV 4.5.1](https://github.com/opencv/opencv/tree/4.5.1) с использованием [дополнительных модулей](https://github.com/opencv/opencv_contrib/tree/4.5.1)  

* [Google gflags 2.2.2](https://github.com/gflags/gflags/tree/v2.2.2)  

* [Google glog 0.4.0](https://github.com/google/glog/tree/v0.4.0)  

* [Open Multiple View Geometry 1.6](https://github.com/openMVG/openMVG/tree/v1.6)  

* [Visualization and Computer Graphics Library 1.0.1](https://github.com/cdcseacave/VCG/tree/v1.0.1)  

* [Open Multiple View Stereovision 1.1.1](https://github.com/cdcseacave/openMVS/tree/v1.1.1)  

* [Eigen 3.3](https://gitlab.com/libeigen/eigen/-/tree/3.3)  

* [Librdkafka 1.5.3](https://github.com/edenhill/librdkafka/tree/v1.5.3)  

* [CGAL 5.2](https://github.com/CGAL/cgal/tree/v5.2)

* [Amazon Web Services SDK 1.8.161](https://github.com/aws/aws-sdk-cpp/tree/1.8.161)  

Подробное описание зависимостей и функционала, для которого данные зависимости используются, можно прочитать [здесь](./doc/dependencies/DependenciesInfo(RU).md).

<a name="modules-desciption"></a>

### 4. Описание модулей проекта  

<a name="deploying"></a>

### 5. Инструкция по развёртыванию  

#### 5.1 Локальное развёртывание  

##### 5.1.1 Локальное развёртывание на ОС Windows  

Локальное развёртывание на ОС Windows в данный момент не поддерживается.

##### 5.1.2 Локальное развёртывание на ОС Linux  

#### 5.2 Развёртывание с использованием Docker  

##### 5.2.1 Развёртывание с использованием Docker на ОС Windows  

В данный момент Nvidia docker не поддерживается на ОС Windows, из-за чего развёртывание с ис использованием Docker на ОС Windows в данный момент не возможно.

##### 5.2.2 Развёртывание с использованием Docker на ОС Linux  

Для развёртывания проекта через Docker необходимо установить пакет `nvidia-docker2`.
Для этого необходимо выполнить следующие команды:  

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
&& sudo apt install nvidia-docker2 \
&& sudo systemctl restart docker
```

Так же для работы сервиса требуется Nvidia драйвер версии 455.28 и выше.  

Установку драйвера можно выполнить несколькими путями.  

1) Зайти на [этот](https://www.nvidia.ru/Download/index.aspx?lang=ru) сайт.  
2) Выбрать свою видеокарту.  
3) Скачать драйвер.  
4) Выполнить установку.  

Так же драйвер можно установить используя apt репозиторий. Для этого необходимо выполнить следующие команды:

```
sudo apt-add-repository ppa:graphics-drivers/ppa \
sudo apt update \
sudo apt install nvidia-driver-455
```



<a name="configuring"></a>

### 6. Конфигурирование  

Для конфигурирования сервиса используются конфигурационные файлы в формате JSON. 

<a name="usage"></a>

### 7. Использование  
