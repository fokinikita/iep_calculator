# IEP Calculator — запуск приложения

## 1. Клонировать репозиторий

```bash
git clone https://github.com/fokinikita/iep_calculator.git
```

## 2. В папке проекта выполнить 
```bash
poetry install
```

## 3. В папке src создать .env файл
образец - default.env
yandex api ключ можно взять тут
https://developer.tech.yandex.ru/services?addapi=true 
выбрать
JavaScript API и HTTP Геокодер

## 4. В папке src выполнить 
```bash
poetry run uvicorn apps.api:app --host 127.0.0.1 --port 8000 --reload
```

## 5. Открыть в браузере
http://127.0.0.1:8000/
