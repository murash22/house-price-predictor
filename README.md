## Housing Price Prediction Model

Лекции и тех задание на семинары доступны на [Я.диске](https://disk.yandex.ru/d/vDb3HPumZ2xK0w)  

### Описание проекта
Проект направлен на создание модели машинного обучения для прогнозирования цен на жилье. Модель использует различные характеристики объектов недвижимости для предсказания их рыночной стоимости.

### Структура проекта
```
house-price-prediction/
├── data/
│   ├── raw/                # Исходные данные
│   ├── processed/          # Обработанные данные
├── models/                 # Обученные модели
├── notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb
│   └── train.ipynb
│   service/
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│   └── flask.log
├── src/                   
│   ├── cian_parse.py
│   └── lifecycle.py           
├── requirements.txt       # Требования к зависимостям
└── README.md
```

### Данные
Используемые данные включают следующие характеристики:
* Площадь жилья

### Как запустить
1. Клонируйте репозиторий:
```bash
git clone https://github.com/murash22/house-price-predictor.git
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите Jupyter Notebook:
```bash
python service/app.py
```

### Модели машинного обучения
* **Linear Regression** - базовая линейная регрессия

### Метрики оценки
* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

### Результаты
После обучения модели достигаются следующие результаты:
* MAE: ~ 8774974 руб.
* MSE: ~ 166750 млрд. руб
* RMSE: ~ 12913 тыс. руб.
* R² Score: ~ 0.660649

### Как использовать модель
1. Загрузите данные в формате CSV
2. Обработайте данные с помощью предобработчиков
3. Загрузите обученную модель
4. Сделайте предсказания

### Команда
* **Data Scientist**: [Имя Фамилия]
* **ML Engineer**: [Имя Фамилия]
* **Product Manager**: [Имя Фамилия]

### Лицензирование
Этот проект распространяется под лицензией MIT. Смотрите файл LICENSE для деталей.

### Контакты
Для вопросов и предложений обращайтесь:
* Email: your.email@example.com
* GitHub: @yourusername
* LinkedIn: linkedin.com/in/yourusername
