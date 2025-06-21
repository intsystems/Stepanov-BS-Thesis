# Модель для аугментации изображений

Модель для автоматической аугментации изображений
## 🛠 Установка

1. Клонируйте репозиторий, установите окружение, скачайте нужные зависимости и веса:
```bash
git clone https://github.com/ILIAHHne63/PaperFramework](https://github.com/intsystems/Stepanov-BS-Thesis.git
cd code

conda create --name aug_model python=3.10
conda activate aug_model

pip install -r requirements.txt
cp -r src/alpha-clip/* "AugModel/models/AlphaCLIP/"
```

2. **Скачайте файл модели** по ссылке:  
   [https://drive.google.com/file/d/1e83wWQh9Tsficx0HM2rmc2KyYfBMpumf/view](https://drive.google.com/file/d/1e83wWQh9Tsficx0HM2rmc2KyYfBMpumf/view)

3. **Разместите файл в директории**:
   
   AugmentationModel/AugModel/models/checkpoints/

### Далее можете использовать Example.ipynb
