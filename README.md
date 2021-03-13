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

<a name="modules-desciption"></a>

### 4. Описание модулей проекта  

<a name="deploying"></a>

### 5. Инструкция по развёртыванию  

#### 5.1 Локальное развёртывание  

##### 5.1.1 Локальное развёртывание на ОС Windows  

##### 5.1.2 Локальное развёртывание на ОС Linux  

#### 5.2 Развёртывание с использованием Docker  

##### 5.2.1 Развёртывание с использованием Docker на ОС Windows  

##### 5.2.2 Развёртывание с использованием Docker на ОС Linux  

Для развёртывания проекта через Docker необходимо установить пакет `nvidia-docker2`.
Для этого необходимо выполнить следующие команды:  

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
  

<a name="configuring"></a>

### 6. Конфигурирование  

<a name="usage"></a>

### 7. Использование  
