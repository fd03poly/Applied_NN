# -*- coding: utf-8 -*-
"""applied_nn_project

# ***Измерение визуальной метафоры в пространстве CLIP***

## **Цель проекта**
Исследовать, насколько мультимодальная модель CLIP способна выявлять и отражать семантические связи между компонентами концептуальных метафор в визуальном и текстовом пространствах представления.

## ***Задачи проекта***

1. Подготовить таблицы с текстами и изображениями
2. Загрузить CLIP
3. Получить текстовые эмбеддинги
4. Получить визуальные эмбеддинги
5. Сохранить эмбеддинги в .npy
6. Посчитать cosine similarity
7. Сравнить группы изображений
8. Сравнить русский и английский
9. Построить UMAP
10. Проверить векторную арифметику
11. Проанализировать результаты

## **Темы курса, использованные в проекте**

* Векторные представления в NLP — использованы эмбеддинги текстов и изображений, полученные с помощью модели CLIP, и анализ их семантической близости.
* Введение в CV (Computer Vision) — выполнен анализ визуальных данных и обработка изображений с помощью нейросетевой модели.
* Multimodal Data — исследование основано на совместном анализе текстовых и визуальных представлений в едином семантическом пространстве.
* Введение в нейронные сети — использована предобученная нейросетевая архитектура CLIP для получения эмбеддингов и анализа концептуальных метафор.

# ***Гипотезы исследования***

**H1.** Метафорические выражения имеют более высокое cosine similarity с изображениями, связанными с соответствующей метафорой (source и target), чем с контрольными изображениями (control).

**H2.** Target-фразы имеют более высокое cosine similarity с изображениями target-домена, чем с изображениями source-домена.

**H3.** Сила метафорического эффекта в пространстве эмбеддингов CLIP различается для русскоязычных и англоязычных текстов.

**H4.** Метафорический эффект будет различаться в зависимости от конкретной метафоры.

**H5.** С помощью векторной арифметики можно сдвинуть абстрактный target-концепт в сторону source-области соответствующей метафоры.

# ***Изображения***

**3 типа картинок:**

1. **SOURCE** - это источник метафоры. Это конкретные, физические, визуальные объекты

Например: для TIME IS MONEY

source = money

money:
- coins
- banknotes
- wallet
- cash

2. **TARGET** - это абстрактный концепт

target = time

time:
- clock
- calendar
- alarm

3. **CONTROL** - это рандомные контрольные картинки, они НЕ связаны с метафорой

Картинки выбирали вручную с сайтов: [Unsplash](https://unsplash.com/), [Pexels](https://www.pexels.com/ru-ru/)

# ***Текстовые данные***

[Книга](https://psv4.userapi.com/s/v1/d/RlSBE9Aj7YDuxIa4xbbq80SVq27qT4EHeZwn0eruiKAgOOW2EOxd_s5BacEej6JDuMoEwmBM4LoLGJoef66IHWiLQmd5Va5qAwB33Boc3xQFURTLxMbykw/lakoff_Metafory_kotorymi_my_zhivem.pdf) из которой мы брали метафоры Лакоффа и Джонсона на английском и на русском

Метафоры Джорджа Лакоффа и Марка Джонсона — это концепция из их основополагающей книги «Метафоры, которыми мы живём» (1980 год). Авторы доказали, что метафора — это не просто красивый литературный приём, а фундаментальный механизм мышления, с помощью которого мы понимаем сложные абстрактные вещи через призму физического опыта.

[Наш датасет](https://docs.google.com/spreadsheets/d/1TZ-XjBtrFGRnqNI8PNRwGjl9HbsTRa1DUUbUneZ3vOM/edit?usp=sharing) и описание:

| Переменная | Что означает | Пример |
|------------|--------------|--------|
| `metaphor` | Название метафоры | `"time is money"` |
| `target_domain` | **Целевая область** — абстрактное понятие, которое описываем | `time`, `love`, `argument` |
| `source_domain` | **Источниковая область** — конкретное понятие, через которое описываем | `money`, `journey`, `war` |
| `target_phrase_1-5` | Фразы с **метафорическим** использованием целевой области | `"saving time"`, `"wasting time"` |
| `source_phrase_1-5` | Фразы с **буквальным** использованием источниковой области | `"saving money"`, `"wasting money"` |
| `pictures (link)` | Ссылка на папку с изображениями для этой метафоры | https://github.com/fd03poly/Applied_NN/tree/main/pictures/time%20is%20money |

# ***Методы***

1. **Подготовка данных**: Отбор и структурирование текстовых фраз и изображений *(source, target, control)*.

2. **Извлечение эмбеддингов** (CLIP): Использование мультимодальной нейросети openai/clip-vit-base-patch32 для перевода текстов и изображений в единое семантическое векторное пространство.

3. **Cosine Similarity**: Вычисление косинусного сходства между текстовыми эмбеддингами и визуальными эмбеддингами для оценки того, насколько текст "близок" к той или иной визуальной группе.

4. **Визуализация** (UMAP): Понижение размерности векторов для визуального анализа того, как тексты и изображения кластеризуются в пространстве CLIP.

5.  **Векторная арифметика**: Проверка гипотезы о семантическом сдвиге с помощью операции target - abstract_anchor + source. Цель — понять, сместит ли эта операция вектор в сторону метафорической области-источника (source).

# ***Импорты и загрузка данных***
"""

!pip install umap-learn

import os
from pathlib import Path # работа с путями

import numpy as np
import pandas as pd

import torch
from PIL import Image # работа с картинками
from transformers import CLIPProcessor, CLIPModel

import umap

import matplotlib.pyplot as plt
import seaborn as sns

# Commented out IPython magic to ensure Python compatibility.
# Доступ к гитхабу

!git clone https://github.com/fd03poly/Applied_NN.git
# %cd Applied_NN

!mv "/content/Applied_NN/pictures/time is money/tartget" "/content/Applied_NN/pictures/time is money/target" # опечатка на гитхабе

!git config --global user.email "fd03poly@gmail.com"
!git config --global user.name "fd03poly"

TOKEN = "..."
!git remote set-url origin https://fd03poly:{TOKEN}@github.com/fd03poly/Applied_NN.git
!git remote -v

!pwd
!ls
!ls /content/Applied_NN

Path("results").mkdir(exist_ok=True)
Path("results/figures").mkdir(exist_ok=True)
Path("results/tables").mkdir(exist_ok=True)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") #делает данные понятными для модели (из слов в числа, из картинок в тензор нужного формата)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval() #не обучаем, только используем модель

TEXTS_DIR = Path("/content/Applied_NN/texts")

RU_TEXTS_PATH = TEXTS_DIR / "texts_ru.csv"
EN_TEXTS_PATH = TEXTS_DIR / "texts_en.csv"

texts_ru_wide = pd.read_csv(RU_TEXTS_PATH)
texts_en_wide = pd.read_csv(EN_TEXTS_PATH)

texts_ru_wide

"""# ***Тексты***


текст -> processor -> text encoder -> text projection -> эмбеддинг
"""

# это одни и те же метафоры, но на разных яхыках. Но python видит две разные строки. Поэтому создаётся единый ID

metaphor_id_map = {
    "время — это деньги": "time_money",
    "любовь — это путешествие": "love_journey",
    "спор — это война": "argument_war",

    "time is money": "time_money",
    "love is a journey": "love_journey",
    "argument is war": "argument_war",
}

# меняем формат (wide -> long)

def make_texts_long(df_wide, language):
    rows = []

    for _, row in df_wide.iterrows(): # берём каждую строку таблицы и ищем IDs
        metaphor = row["metaphor"]
        metaphor_id = metaphor_id_map.get(metaphor, metaphor)

        # добавляем саму метафору
        rows.append({
            "metaphor_id": metaphor_id,
            "language": language,
            "text": row["metaphor"],
            "text_type": "metaphor",
            "pair_id": "none"
        })

        # добавляем target domain
        rows.append({
            "metaphor_id": metaphor_id,
            "language": language,
            "text": row["target_domain"],
            "text_type": "target_domain",
            "pair_id": "none"
        })

        # добавляем source domain
        rows.append({
            "metaphor_id": metaphor_id,
            "language": language,
            "text": row["source_domain"],
            "text_type": "source_domain",
            "pair_id": "none"
        })

        # phrase columns
        for col in df_wide.columns:
            if col.startswith("target_phrase_"):
                pair_number = col.replace("target_phrase_", "") #создаем пары между фразами (экономить время <-> экономить деньги)
                source_col = f"source_phrase_{pair_number}"

                target_text = row[col]

                if pd.notna(target_text):
                    rows.append({
                        "metaphor_id": metaphor_id,
                        "language": language,
                        "text": target_text,
                        "text_type": "target_phrase",
                        "pair_id": f"p{pair_number}"
                    })

                if source_col in df_wide.columns:
                    source_text = row[source_col]

                    if pd.notna(source_text):
                        rows.append({
                            "metaphor_id": metaphor_id,
                            "language": language,
                            "text": source_text,
                            "text_type": "source_phrase",
                            "pair_id": f"p{pair_number}"
                        })

    df_long = pd.DataFrame(rows)

    return df_long

texts_ru_long = make_texts_long(texts_ru_wide, language="ru")
texts_en_long = make_texts_long(texts_en_wide, language="en")

texts_metadata = pd.concat(
    [texts_ru_long, texts_en_long],
    ignore_index=True
) # объединяем русский и английский

texts_metadata.insert(
    0,
    "text_id",
    [f"txt_{i:04d}" for i in range(len(texts_metadata))]
)  # создаём уникальные ID для каждой фразы (потом текст эмбеддинг можно связать с текстом)

texts_metadata.head()

texts = texts_metadata["text"].tolist() #все тексты списком

text_embeddings = []

with torch.no_grad(): #без вычисления градиентов (не готовит модель к обучению)
    for text in texts: #берем по очереди каждый текст
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ) #превращает текст в числа

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        text_outputs = model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ) # текст проходит через текстовый трансформер CLIP.

        pooled_output = text_outputs.pooler_output # свертка векторов (было несколько для каждого слова -> стал один для всей фразы)

        text_features = model.text_projection(pooled_output) # переводит оба типа данных (картинки и текст) в одно пространство (тогда можно считать cosine similarity)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # нормализация - длина вектора 1

        text_embeddings.append(text_features.cpu().numpy()[0]) #NumPy не умеет работать с GPU-тензорами

text_embeddings = np.vstack(text_embeddings) #из списка массивов в одну матрицу

print(text_embeddings.shape)

#Path("data").mkdir(exist_ok=True)
#Path("embeddings").mkdir(exist_ok=True)

# сохраняем таблицу с информацией о текстах
#texts_metadata.to_csv("data/texts_metadata.csv", index=False)

# сохраняем эмбеддинги текстов
#np.save("embeddings/text_embeddings.npy", text_embeddings)

# сохраняем id текстов в том же порядке, что и эмбеддинги
#np.save("embeddings/text_ids.npy", texts_metadata["text_id"].values)

"""# ***Картинки***

картинка -> processor -> vision encoder -> visual projection -> эмбеддинг
"""

PICTURES_DIR = Path("/content/Applied_NN/pictures")

metaphor_folders = {
    "time is money": "time_money",
    "argument is war": "argument_war",
    "love is a journey": "love_journey",
} # делаем короткие идентификаторы папок

image_extensions = {".jpg"} #у нас картики только в этом формате

rows = []

for folder_name, metaphor_id in metaphor_folders.items():
    for image_group in ["source", "target"]: # для каждой метафоры берём две папки по итерациям
        folder_path = PICTURES_DIR / folder_name / image_group # cобираем путь

        for image_path in folder_path.iterdir(): # присваивает каждой картинке номер
            if image_path.suffix.lower() in image_extensions: # берем только jpg
                rows.append({ # добавляем изображение в таблицу
                    "image_id": f"img_{len(rows):04d}", #уникальнй ID
                    "metaphor_id": metaphor_id,
                    "image_group": image_group,
                    "image_path": str(image_path),
                    "folder_name": folder_name
                })

# общая папка control
control_folder = PICTURES_DIR / "control"

for image_path in control_folder.iterdir():
    if image_path.suffix.lower() in image_extensions:
        rows.append({
            "image_id": f"img_{len(rows):04d}",
            "metaphor_id": "control",
            "image_group": "control",
            "image_path": str(image_path),
            "folder_name": "control"
        })

images_metadata = pd.DataFrame(rows) # cписок словарей превращается в таблицу Pandas

images_metadata

path = images_metadata.iloc[64]["image_path"]
image = Image.open(path)

plt.imshow(image)
plt.axis("off")
plt.show()

images_metadata.groupby(["metaphor_id", "image_group"]).size()

image_embeddings = []

with torch.no_grad(): #отключаем градиенты
    for path in images_metadata["image_path"]:
        image = Image.open(path).convert("RGB") #CLIP ожидает такой формат

        inputs = processor(
            images=image,
            return_tensors="pt"
        ) # jpg -> тензор (размер 224*224 потому что CLIP обучался именно на таком размере, процессор автоматически масштабирует размер)

        pixel_values = inputs["pixel_values"].to(device) # берем пиксели и получаем числовое представление изображения
        vision_outputs = model.vision_model(pixel_values=pixel_values) #Vision Transformer анализирует содержание картинки и превращает его в число
        pooled_output = vision_outputs.pooler_output  #один общий вектор
        image_features = model.visual_projection(pooled_output) # проецируем в общее CLIP-пространство
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # нормализация для cosine similarity

        image_embeddings.append(image_features.cpu().numpy()[0])

image_embeddings = np.vstack(image_embeddings) #из списка в матрицу

print(image_embeddings.shape)

"""70 картинок

эмбеддинг из 512 чисел (У CLIP размерность вектора: 512 чисел. То есть каждый текст и каждая картинка — это точка в пространстве из 512 измерений.)
"""

pooled_output.shape

"""1 картинка

768 признаков
"""

pixel_values.shape

"""1 картинка

3 цветовых канала

224 пикселя

224 пикселя
"""

#Path("embeddings").mkdir(exist_ok=True)
#Path("data").mkdir(exist_ok=True)

# сохраняем таблицу с информацией о картинках
#images_metadata.to_csv("data/images_metadata.csv", index=False)

# сохраняем эмбеддинги картинок
#np.save("embeddings/image_embeddings.npy", image_embeddings)

# сохраняем id картинок в том же порядке, что и эмбеддинги
#np.save("embeddings/image_ids.npy", images_metadata["image_id"].values)

# Проверяем количество

print(len(images_metadata), image_embeddings.shape[0])
print(len(texts_metadata), text_embeddings.shape[0])

Path("results_tables").mkdir(exist_ok=True)

"""# ***Сosine similarity***

$$
\text{Cosine Similarity}(\mathbf{a}, \mathbf{b}) =
\frac{\sum_{i=1}^{n} a_i b_i}
{\sqrt{\sum_{i=1}^{n} a_i^2}\sqrt{\sum_{i=1}^{n} b_i^2}}
$$


Где:

- $\sum_{i=1}^{n} a_i b_i$ — скалярное произведение векторов.
- $\sqrt{\sum_{i=1}^{n} a_i^2}$ и $\sqrt{\sum_{i=1}^{n} b_i^2}$ — длины (нормы) векторов $\mathbf{a}$ и $\mathbf{b}$ соответственно.

Векторная форма записи:

$$
\text{Cosine Similarity}(\mathbf{a}, \mathbf{b})
=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|\,\|\mathbf{b}\|}
$$

**Косинусная близость** — это мера схожести между двумя векторами, определяющая косинус угла между ними. Она показывает, насколько два вектора сонаправлены в многомерном пространстве. Ее главное преимущество — независимость от длины векторов, что важно при сравнении, например, текстов разной длины.

Значение лежит в диапазоне от \(-1\) (полная противоположность) до \(1\) (полное совпадение)

Для нормализованных векторов: скалярное произведение = cosine similarity
"""

# @ - матричное умножение
similarity_matrix = text_embeddings @ image_embeddings.T #автоматически вычисляет cosine similarity между ВСЕМИ текстами и ВСЕМИ картинками.

print(similarity_matrix.shape)

"""78 текстов

70 картинок
"""

# Делаем таблицу всех text-image пар

metaphor_ids = ["time_money", "argument_war", "love_journey"]

rows = []

for i, text_row in texts_metadata.iterrows(): # перебираем все тексты
    text_metaphor = text_row["metaphor_id"] # смотрим к какому типу метафор она относится

    for j, image_row in images_metadata.iterrows(): # перебираем все картики
        image_metaphor = image_row["metaphor_id"] # получаем метафору для картинки
        image_group = image_row["image_group"] # получаем группу для картинки (source, target, control)

        if (text_metaphor == image_metaphor) or (image_group == "control"): # оставляем только текст под картинку и картинку под текст, control остается всегда
            rows.append({
                "text_id": text_row["text_id"],
                "text": text_row["text"],
                "language": text_row["language"],
                "text_type": text_row["text_type"],
                "text_metaphor_id": text_metaphor,
                "pair_id": text_row.get("pair_id", "none"),

                "image_id": image_row["image_id"],
                "image_metaphor_id": image_metaphor,
                "image_group": image_group,
                "image_path": image_row["image_path"],

                "similarity": similarity_matrix[i, j] # берётся число из матрицы сходства
            })

similarity_relevant = pd.DataFrame(rows)
similarity_relevant

best_row = similarity_relevant.loc[similarity_relevant["similarity"].idxmax()]
worst_row = similarity_relevant.loc[similarity_relevant["similarity"].idxmin()]

fig, axes = plt.subplots(1, 2, figsize=(10,5))

# best
img1 = Image.open(best_row["image_path"])
axes[0].imshow(img1)
axes[0].set_title(f"MAX\n{best_row['similarity']:.3f}\n{best_row['text']}")
axes[0].axis("off")

# worst
img2 = Image.open(worst_row["image_path"])
axes[1].imshow(img2)
axes[1].set_title(f"MIN\n{worst_row['similarity']:.3f}\n{worst_row['text']}")
axes[1].axis("off")

plt.show()

target_domain_df = similarity_relevant[
    similarity_relevant["text_type"] == "target_domain"
].copy() # оставляем ТОЛЬКО абстрактные концепты (time, love, argument)

target_domain_summary = (
    target_domain_df
    .groupby([
        "text_metaphor_id",
        "language",
        "text",
        "image_group"
    ])["similarity"]
    .mean()
    .reset_index()
) #усредняем их все

target_domain_summary

target_domain_summary.to_csv(
    "results/tables/target_domain_summary.csv",
    index=False
)

"""* средняя cosine similarity между эмбеддингом текста “love” и ВСЕМИ картинками target-группы = 0.241804"""

target_domain_wide = target_domain_summary.pivot_table(
    index=["text_metaphor_id", "language", "text"],
    columns="image_group",
    values="similarity"
).reset_index() # переворачиваем для удобства

target_domain_wide

"""CLIP чуть лучше связывает "argument" на английском с реальными спорми, а не с войной (0,21), при этом на русском все примерно одинаково.

Тоже самое с "Love", CLIP находит связь между словом love и картинками, которые относятся к любви лучше чем к пути.

CLIP смильнее всего связывает "time" с изображениями времени (часы, календарь), чем с деньгами.

**source_minus_control** — абстрактное понятие ближе к метафорической source-области, чем к контролю? (есть ли у CLIP вообще реакция на метафорический источник (деньги, война, путешествия)) *source_minus_control > 0, значит time ближе к картинкам денег, чем к контрольным картинкам.*

**target_minus_control** — показывает, насколько хорошо CLIP связывает понятие с его прямыми визуальными образами (насколько абстрактный концепт реально распознаётся через визуальные образы) *(хорошо - это больше нуля)*

**source_minus_target** — Это сравнение метафорической области с прямой областью (что ближе к тексту — метафора или буквальный смысл) *(хорошо - это меньше нуля)*
"""

target_domain_wide["source_minus_control"] = (
    target_domain_wide["source"] - target_domain_wide["control"]
)

target_domain_wide["target_minus_control"] = (
    target_domain_wide["target"] - target_domain_wide["control"]
)

target_domain_wide["source_minus_target"] = (
    target_domain_wide["source"] - target_domain_wide["target"]
)

target_domain_wide

"""На английском модель воспринимает "argument" как реальные сцены споров, а не как метафору войны. При этом на русском метафора практически не кодируется (слабая семантическая структура, все почти на уровне шума).

CLIP на английском интерпретирует "love" через визуальные сценарии движения/путешествия (target-control = +0.055). На русском же всё маленькое и положительное, то есть слабый, но стабильный результат

Модель интерпретирует "time" буквально, а не метафорически (target-control = +0.069)

**ВЫВОД**
1. CLIP не следует метафорам Лакоффа и Джонсона. Модель предпочитает буквальную визуальную семантику.
2. Метафорический источник почти не выделяется как отдельная семантика.
3. Русский язык слабее структурирован, так как CLIP обучен в основном на англоязычных данных

"""

phrase_df = similarity_relevant[
    similarity_relevant["text_type"] == "target_phrase" # берём только строки, где текст — это метафорическая фраза.
].copy()

phrase_summary = (
    phrase_df
    .groupby([
        "text_metaphor_id",
        "language",
        "pair_id",
        "text",
        "image_group"
    ])["similarity"]
    .mean()
    .reset_index()
) # для каждой фразы считаем среднюю близость ко всем картинкам группы.

phrase_wide = phrase_summary.pivot_table(
    index=["text_metaphor_id", "language", "pair_id", "text"],
    columns="image_group",
    values="similarity"
).reset_index() #переворачиваем

# Метaфорические (source) картинки лучше случайных?
phrase_wide["source_minus_control"] = (
    phrase_wide["source"] - phrase_wide["control"]
)

# Буквальные (target) картинки лучше случайных?
phrase_wide["target_minus_control"] = (
    phrase_wide["target"] - phrase_wide["control"]
)

# Что ближе к фразе?
phrase_wide["source_minus_target"] = (
    phrase_wide["source"] - phrase_wide["target"]
)


phrase_wide

phrase_wide.to_csv(
    "results/tables/phrase_wide.csv",
    index=False
)

"""CLIP воспринимает defending an argument и attacking a point одновременно и как спор, и как защиту в войне, при этом winning an argument (source-target = -0.029) здесь target значительно выше, то есть CLIP воспринимает это скорее как спор, чем как войну. Получается, что английские выражения действительно демонстрируют признаки метафорического кодирования.

Во всех случаях target > source. То есть CLIP связывает выражения сильнее с картинками путешествия. Напимер, relationship is not going anywhere буквально содержит идею движения. Поэтому модель видит в нём "путешествие".

saving time, wasting time, running out of time - CLIP значительно сильнее связывает фразу с target (time). Investing time source-target = +0.001 (практически ноль). Это означает CLIP одинаково связывает выражение и с временем, и с деньгами. Потому что слово investing само происходит из финансового домена.

---

Разница между русским и английским примерно в несколько раз. Это говорит о том, что CLIP гораздо лучше распознаёт метафорические связи на английском языке.

Это может происходить потому что:

- модель обучалась преимущественно на английских данных;
- английские метафорические конструкции чаще встречались в обучающем корпусе;
- русские фразы представлены в модели слабее.


---


Самый сильный target-эффект - saving time (target-control = 0.069)

Самый сильный метафорический баланс - defending an argument (source-target ≈ 0)

Лучший пример LOVE IS A JOURNEY - relationship is not going anywhere

Лучший пример TIME IS MONEY - investing time


"""

# Что происходит в среднем по всей метафоре, а не по отдельным фразам?

phrase_effects_summary = (
    phrase_wide
    .groupby(["text_metaphor_id", "language"]) #сгруппируй все фразы одной метафоры и одного языка вместе
    [["source_minus_control", "target_minus_control", "source_minus_target"]]
    .mean()
    .reset_index()
)

phrase_effects_summary

phrase_effects_summary.to_csv(
    "results/tables/phrase_effects_summary.csv",
    index=False
)

"""**H1.** Метафорические выражения имеют более высокое cosine similarity с изображениями, связанными с соответствующей метафорой (source и target), чем с контрольными изображениями (control). - **Гипотеза поддерживается.**

**H2.** Target-фразы имеют более высокое cosine similarity с изображениями target-домена, чем с изображениями source-домена. - **Гипотеза поддерживается.**

**H3.** Сила метафорического эффекта различается для русского и английского языков - **Гипотеза поддерживается.**

Argument is war - все группы показывают значения выше чем контрольные картинки значит CLIP действительно видит смысловую связь. Но target > source. Значит, CLIP воспринимает метафорические выражения про спор и через войну, и через буквальные сцены спора, но буквальный смысл немного сильнее. Для русского языка модель почти не различает source и target.

Love is a journey - здесь разница большая, то есть target значительно лучше source. CLIP очень хорошо связывает фразы про отношения с изображениями отношений/движения. Но метафорический источник (journey) слабее. То есть модель понимает выражения, но интерпретирует их скорее буквально, чем через концептуальную метафору. С русским ситуация такая же.

Time is money - Фразы saving time, wasting time, losing time намного сильнее тянутся к target-картинкам. CLIP интерпретирует такие выражения прежде всего через буквальное понятие времени. На русском source > target, но разница очень маленькая, поэтому серьёзных выводов делать нельзя.
"""

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

palette = sns.color_palette("husl", phrase_df["image_group"].nunique())

# Настраиваем размер шрифтов
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Создаем график
g = sns.catplot(
    data=phrase_df,
    x="image_group",
    y="similarity",
    hue="image_group",
    col="text_metaphor_id",
    row="language",
    kind="box",
    height=5,
    aspect=1.3,
    sharey=False,
    palette=palette,
    linewidth=2,
    fliersize=5,
    boxprops=dict(alpha=0.85, linewidth=2),
    medianprops=dict(color='black', linewidth=2.5),
    whiskerprops=dict(linewidth=2),
    capprops=dict(linewidth=2),
    showfliers=True
)

for ax in g.axes.flat:
    ax.set_xlabel("Image Group", fontsize=12, labelpad=10)
    ax.set_ylabel("Cosine Similarity", fontsize=12, labelpad=10)

    # Добавляем легенду на каждый график
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Удаляем старую легенду
        for legend in ax.legend_:
            legend.remove()
        ax.legend(title="Image Groups", loc="best", fontsize=9, title_fontsize=10)

# Заголовки подграфиков
g.set_titles("{row_name} | Metaphor {col_name}", size=14, pad=10, fontweight='semibold')

# Отступы между подграфиками
g.fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Общий заголовок
g.fig.suptitle('Distribution of Cosine Similarity by Image Groups\nfor Target Phrase Metaphors',
               fontsize=17, fontweight='bold', y=1.10)

# Сохраняем
plt.savefig( "results/figures/cosine_similarity_boxplot.png", dpi=300, bbox_inches="tight" )

plt.show()

"""**H4.** Метафорический эффект будет различаться в зависимости от конкретной метафоры. - **Гипотеза подтверждается**

Мы смотрим, насколько фразы вроде *saving time / экономить время, defending an argument / защищать аргумент, relationship path / путь отношений* похожи на изображения source, target и control.

Во всех шести случаях control в среднем ниже, чем source и target. Это значит, что CLIP отличает релевантные изображения от контрольныз: target-фразы не просто случайно похожи на любые картинки, а действительно сильнее связаны с тематически подходящими визуальными областями.

При этом почти везде target выше source, особенно в русском. Это значит, что прямая связь фразы с target-областью обычно сильнее, чем метафорическая связь с source-областью. Иными словами, CLIP лучше связывает фразы с их буквальной/основной темой, чем с метафорическим образом.


---



Для английских фраз метафоры **time is money** видно, что source примерно сопоставим или чуть выше target. Это показывает, что английские time-фразы достаточно хорошо связаны не только с изображениями времени, но и с изображениями денег. То есть метафорическая область money проявляется довольно заметно. Можно интерпретировать, что в CLIP-пространстве частично активируют экономическую область.

Английские фразы про **любовь/отношения** одновременно связаны и с прямой областью отношений, и с областью пути/путешествия. Разница между source и target небольшая.

На английском, фразы про **спор** связаны не только с изображениями дискуссии, но и с военной областью. Однако разница между source и target небольшая, поэтому нельзя сказать, что военная область доминирует.


---

Русские фразы типа экономить время, тратить время, вкладывать время сильнее связаны с target-изображениями времени, чем с изображениями денег. Но source всё равно выше control, значит связь с деньгами тоже есть.Получается что для русского **time is money** метафорическая связь проявляется, но прямая связь с областью времени сильнее.

Русские фразы про **любовь/отношения** сильнее всего связаны с изображениями любви/отношений. Связь с путешествием/путём есть, но она слабее прямой target-связи.

Русские фразы про спор лучше всего связываются с изображениями собстыенно спора, но при этом изображения войны всё равно ближе, чем контрольные изображения. Метафорический эффект есть, но он слабее прямой тематической связи.

Главное различие такое:
- в английском source и target часто близки друг к другу;
- в русском target обычно заметно выше source.
"""

# Устанавливаем стиль
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Используем ту же палитру husl
palette = sns.color_palette("husl", phrase_wide["language"].nunique())

# Настраиваем размер шрифтов
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Создаем график
plt.figure(figsize=(14, 7))

sns.stripplot(
    data=phrase_wide,
    x="text_metaphor_id",
    y="source_minus_target",
    hue="language",
    dodge=True,
    jitter=0.25,
    size=10,  # Увеличенный размер точек
    palette=palette,
    edgecolor="black",  # Черная обводка для лучшей видимости
    linewidth=1.5  # Толщина обводки
)

# Добавляем горизонтальную линию на нуле
plt.axhline(0, linestyle="--", color="red", linewidth=2, alpha=0.7)

# Подписи
plt.title("Source-Target Effect for Individual Target Phrases",
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Metaphor", fontsize=12, labelpad=10, fontweight='semibold')
plt.ylabel("Source - Target", fontsize=12, labelpad=10, fontweight='semibold')

# Поворот подписей на оси X
plt.xticks(rotation=20, fontsize=10)
plt.yticks(fontsize=10)

# Добавляем легенду
plt.legend(title="Language", title_fontsize=11, fontsize=10, loc="best")

# Улучшаем компоновку
plt.tight_layout()

# Сохраняем график
plt.savefig( "results/figures/source_control_effect_stripplot.png", dpi=300, bbox_inches="tight" )

plt.show()

"""График показывает разницу между сходством с source- и target-изображениями для каждой отдельной фразы.

Английские фразы типа **winning an argument, losing an argument, demolishing an argument** CLIP связывает сильнее с изображениями спора (target), чем с изображениями войны (source). То есть модель предпочитает буквальный целевой домен.

Фразы про **отношения** на английском гораздо сильнее ассоциируются с изображениями отношений, чем с изображениями путешествий - это достаточно устойчивый эффект.


Английские фразы типа **saving time, running out of time, losing time** CLIP намного сильнее связывает с изображениями времени, чем с изображениями денег.


---

Для русских метафор CLIP демонстрирует сопоставимое сходство с изображениями source- и target-доменов, не отдавая явного предпочтения одному из них. В отличие от этого, для английских метафор наблюдается более выраженное предпочтение изображений target-домена.

# ***UMAP***

= Uniform Manifold Approximation and Projection (равномерная аппроксимация многообразия и проекция) — метод, который берёт данные с большим числом признаков (которые мы получили после CLIP) и проецирует их в 2D так, чтобы их можно было визуализировать.

Мы используем UMAP, что сжать пространство эмбеддингов (text_embeddings + image_embeddings) до двух координат:
эмбеддинги → UMAP → точка на плоскости. При этом схожие объекты на графике окажутся рядом друг с другом.

Что конкретно мы хотим проверить:

1. Лежат ли тексты и картинки одной метафоры рядом?
2. Отделяются ли source / target / control?
3. Лежат ли русские и английские тексты группы одной метафоры близко друг к другу?
4. Насколько сильно отделяются текстовые и визуальные эмбеддинги?

То есть в целом с помощью UMAP мы хотим визуально оценить, располагаются ли тексты и изображения, относящиеся к одной концептуальной метафоре, рядом друг с другом в общем семантическом пространстве.
"""

# нормализуем эмбеддинги перед UMAP

text_embeddings_norm = text_embeddings / np.linalg.norm( #считает длину каждого вектора.
    text_embeddings, axis=1, keepdims=True
)

image_embeddings_norm = image_embeddings / np.linalg.norm(
    image_embeddings, axis=1, keepdims=True
)

# объединяем текстовые и визуальные эмбеддинги (складываеем кол-во картинок с кол-вом текстов (70+78 = 148))
all_embeddings = np.vstack([
    text_embeddings_norm,
    image_embeddings_norm
])

print(all_embeddings.shape)

# Готовим metadata для UMAP

# metadata для текстов
text_umap = texts_metadata.copy()
text_umap["modality"] = "text"
text_umap["group"] = text_umap["text_type"]
text_umap["label"] = text_umap["text"]

# metadata для изображений
image_umap = images_metadata.copy()
image_umap["modality"] = "image"
image_umap["group"] = image_umap["image_group"]
image_umap["language"] = "none"
image_umap["label"] = image_umap["image_id"]

common_cols = [
    "metaphor_id",
    "modality",
    "group",
    "language",
    "label"
] #выбираем одинаковые столбцы

umap_all_metadata = pd.concat( #склеиваем их
    [
        text_umap[common_cols],
        image_umap[common_cols]
    ],
    ignore_index=True
)

umap_all_metadata

# Строим UMAP

reducer_all = umap.UMAP(
    n_neighbors=15, #когда UMAP будет решать, куда поставить точку, он будет смотреть примерно на 15 ближайших соседей.
    min_dist=0.3, #насколько можно сжимать кластеры.
    metric="cosine", #UMAP будет использовать ту же меру сходства, что и CLIP
    random_state=42 # чтобы каждый запуск давал одинаковую картинку.
)

coords_all = reducer_all.fit_transform(all_embeddings) # из 512 делаем 2

umap_all_metadata["umap_x"] = coords_all[:, 0]
umap_all_metadata["umap_y"] = coords_all[:, 1]

"""## ***1. Общий UMAP: тексты + изображения***
разделение по языку и по модальности
"""

# стиль
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# палитра husl для метафор
metaphors = umap_all_metadata["metaphor_id"].unique()
palette = dict(
    zip(
        metaphors,
        sns.color_palette("husl", len(metaphors))
    )
)

# создаем 2 панели
fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 7),
    sharex=True,
    sharey=True
)

languages = ["en", "ru"]
titles = ["English", "Russian"]

for ax, lang, title in zip(axes, languages, titles):

    # тексты данного языка
    text_data = umap_all_metadata[
        (umap_all_metadata["modality"] == "text") &
        (umap_all_metadata["language"] == lang)
    ]

    # все изображения
    image_data = umap_all_metadata[
        umap_all_metadata["modality"] == "image"
    ]

    plot_data = pd.concat([text_data, image_data])

    sns.scatterplot(
        data=plot_data,
        x="umap_x",
        y="umap_y",
        hue="metaphor_id",
        style="modality",
        palette=palette,
        s=120,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

# общая подпись
fig.suptitle(
    "UMAP of CLIP Text and Image Embeddings",
    fontsize=20,
    fontweight="bold"
)

plt.savefig( "results/figures/UMAP_Text_and_Image_Embeddings.png", dpi=300, bbox_inches="tight" )


plt.tight_layout()
plt.show()

"""Мы видим, что все изображения находятся слева сверху; все тексты находятся справа.

То есть CLIP в наших данных не смешивает текстовые и визуальные объекты в единое облако. UMAP пытается сохранить глобальную структуру данных. Поэтому он сначала разделил пространство по модальностям изображения отдельно, тексты отдельно.

Английские тексты находятся сверху справа. Русские тексты находятся снизу справа. То есть UMAP идеально разделил английский и русский. Несмотря на использование общего мультимодального пространства CLIP, языковая принадлежность остается сильным фактором организации текстовых эмбеддингов.

Внутри английских текстов видны отдельные метафоры. Видно три отдельные группы. Это означает, что CLIP различает разные концептуальные метафоры. То есть модель понимает, что время, любовь, спор — это разные смысловые области.

Внутри русских текстов метафоры различаются слабее& В русской части цвета смешиваются сильнее, а границы между метафорами менее четкие.

Изображения тоже делятся, хоть и явно хуже, но некоторые картинки очень близко друг к другу.

## ***2. UMAP только текстовых эмбеддингов***

разделение по языку
"""

text_only_metadata = texts_metadata.copy()
text_only_metadata["label"] = text_only_metadata["text"] #подпись точки на графике

reducer_text = umap.UMAP(
    n_neighbors=8,
    min_dist=0.3,
    metric="cosine",
    random_state=42
) #cоздаётся объект UMAP с заданными параметрами (объяснение всего выше)

coords_text = reducer_text.fit_transform(text_embeddings_norm) #из 512 в 2D

text_only_metadata["umap_x"] = coords_text[:, 0] # полученные на прдыдущем шагу координаты записываются в таблицу
text_only_metadata["umap_y"] = coords_text[:, 1]

text_only_metadata

# Устанавливаем стиль
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Используем палитру husl
palette = sns.color_palette("husl", text_only_metadata["metaphor_id"].nunique())

# Настраиваем размер шрифтов
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Создаем график с запасом места для легенды
fig, ax = plt.subplots(figsize=(14, 9))

# Основной scatterplot
scatter = sns.scatterplot(
    data=text_only_metadata,
    x="umap_x",
    y="umap_y",
    hue="metaphor_id",
    style="language",
    s=120,
    alpha=0.9,
    palette=palette,
    edgecolor="black",
    linewidth=0.8,
    ax=ax
)

# Подписи на английском
ax.set_title("UMAP of Text Embeddings: Russian vs English",
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("UMAP 1", fontsize=12, labelpad=10, fontweight='semibold')
ax.set_ylabel("UMAP 2", fontsize=12, labelpad=10, fontweight='semibold')

# УДАЛЯЕМ СТАРУЮ ЛЕГЕНДУ от seaborn
if ax.get_legend():
    ax.get_legend().remove()

# СОЗДАЁМ ОДНУ ОБЪЕДИНЁННУЮ ЛЕГЕНДУ внутри графика
handles, labels = ax.get_legend_handles_labels()
n_hue = text_only_metadata["metaphor_id"].nunique()

# Объединяем обе легенды в одну
all_handles = handles
all_labels = labels

# Добавляем разделители между секциями
legend = ax.legend(
    all_handles,
    all_labels,
    title="Metaphor ID / Language",
    title_fontsize=12,
    fontsize=10,
    loc="upper right",  # Внутри графика
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    edgecolor='gray'
)

# Улучшаем компоновку
plt.tight_layout()

# Сохраняем график с высоким разрешением
plt.savefig( "results/figures/UMAP_text_embeddings_ru_en.png", dpi=300, bbox_inches="tight" )

plt.show()

"""1. Четкое разделение по языкам

* Слева (UMAP 1 < 0): почти все английские тексты
* Справа (UMAP 1 > 5): почти все русские тексты

Это означает, что модель кодирует язык сильнее, чем смысл метафоры. Русские и английские фразы с одинаковым значением оказываются в разных частях пространства.

2. Языковая асимметрия

Английские точки более компактные и плотные, образуют определенные кластеры
Русские точки более смешанные
"""

# Устанавливаем стиль
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Используем палитру husl
palette = sns.color_palette("husl", text_only_metadata["metaphor_id"].nunique())

# Настраиваем размер шрифтов
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Создаем график
fig, ax = plt.subplots(figsize=(16, 11))

# Основной scatterplot
scatter = sns.scatterplot(
    data=text_only_metadata,
    x="umap_x",
    y="umap_y",
    hue="metaphor_id",
    style="language",
    s=100,
    alpha=0.7,
    palette=palette,
    edgecolor="black",
    linewidth=0.8,
    ax=ax
)

# Подписи
ax.set_title("UMAP of Text Embeddings: Russian vs English (with phrase labels)",
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("UMAP 1", fontsize=12, labelpad=10, fontweight='semibold')
ax.set_ylabel("UMAP 2", fontsize=12, labelpad=10, fontweight='semibold')

# Удаляем старую легенду
if ax.get_legend():
    ax.get_legend().remove()

# Добавляем подписи к точкам
for idx, row in text_only_metadata.iterrows():
    # Текст фразы (обрезаем если длинный)
    phrase = str(row.get('text', row.get('label', '')))
    if len(phrase) > 25:
        phrase = phrase[:25] + '...'

    # Определяем смещение подписи в зависимости от языка
    # Русские слева, английские справа - чтобы подписи не выходили за края
    if row['language'] == 'ru':
        ha = 'right'
        x_offset = 0.15
    else:
        ha = 'left'
        x_offset = -0.15

    ax.annotate(
        phrase,
        xy=(row['umap_x'], row['umap_y']),
        xytext=(x_offset, 0),
        textcoords='offset points',
        fontsize=7,
        ha=ha,
        va='center',
        alpha=0.8,
        fontstyle='italic',
        color='#333333'
    )

# Создаем легенду
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles, labels,
    title="Metaphor ID / Language",
    title_fontsize=11,
    fontsize=9,
    loc="lower left",
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95
)

plt.tight_layout()
plt.savefig( "results/figures/UMAP_text_with_labels.png", dpi=300, bbox_inches="tight" )
plt.show()

"""# ***3. UMAP только визуальных эмбеддингов***"""

image_only_metadata = images_metadata.copy()
image_only_metadata["label"] = image_only_metadata["image_id"] #В столбец label записывается идентификатор изображения

reducer_image = umap.UMAP(
    n_neighbors=8,
    min_dist=0.3,
    metric="cosine",
    random_state=42
) #те же параметры

coords_image = reducer_image.fit_transform(image_embeddings_norm) #преобразование в координаты

image_only_metadata["umap_x"] = coords_image[:, 0] #сохраняем координаты в таблицу
image_only_metadata["umap_y"] = coords_image[:, 1]

image_only_metadata

# Устанавливаем стиль
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Используем палитру husl
palette = sns.color_palette("husl", image_only_metadata["metaphor_id"].nunique())

# Настраиваем размер шрифтов
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Создаем график с запасом места для легенды
fig, ax = plt.subplots(figsize=(14, 9))

# Основной scatterplot
scatter = sns.scatterplot(
    data=image_only_metadata,
    x="umap_x",
    y="umap_y",
    hue="metaphor_id",
    style="image_group",
    s=120,
    alpha=0.9,
    palette=palette,
    edgecolor="black",
    linewidth=0.8,
    ax=ax
)

# Подписи на английском
ax.set_title("UMAP of Image Embeddings by Metaphor and Image Group",
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("UMAP 1", fontsize=12, labelpad=10, fontweight='semibold')
ax.set_ylabel("UMAP 2", fontsize=12, labelpad=10, fontweight='semibold')

# Создаем одну объединенную легенду внутри графика
handles, labels = ax.get_legend_handles_labels()

legend = ax.legend(
    handles,
    labels,
    title="Metaphor ID / Image Group",
    title_fontsize=12,
    fontsize=10,
    loc="lower right",
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    edgecolor='gray'
)

# Улучшаем компоновку
plt.tight_layout()

# Сохраняем график с высоким разрешением
plt.savefig( "results/figures/UMAP_image_embeddings.png", dpi=300, bbox_inches="tight" )

plt.show()

#задаем координаты выбросов
outliers_coords = [
    {"name": "Выброс 1: time_money source", "x": -0.2, "y": 12.8, "color": "red"},
    {"name": "Выброс 2: control", "x": -0.3, "y": 6.9, "color": "purple"},
    {"name": "Выброс 3: argument_war target", "x": 1.1, "y": 13.1, "color": "green"},
    {"name": "Выброс 4: argument_war target", "x": 0.6, "y": 5.9, "color": "green"},
    {"name": "Выброс 5: control", "x": 1.9, "y": 10.0, "color": "purple"},
]

# создание соответствия между идентификаторами изображений и путями к файлам
BASE_DIR = "/content/Applied_NN/pictures"
METAPHOR_MAP = {
    "time_money": "time is money",
    "love_journey": "love is a journey",
    "argument_war": "argument is war",
    "control": "control"
}

# Порядок папок (алфавитный)
folders_order = [
    ("argument_war", "source"),
    ("argument_war", "target"),
    ("control", None),
    ("love_journey", "source"),
    ("love_journey", "target"),
    ("time_money", "source"),
    ("time_money", "target"),
]

label_to_path = {}
global_idx = 0

for metaphor_id, group in folders_order:
    metaphor_folder = METAPHOR_MAP[metaphor_id]
    if group is None:
        folder_path = os.path.join(BASE_DIR, metaphor_folder)
    else:
        folder_path = os.path.join(BASE_DIR, metaphor_folder, group)

    if not os.path.exists(folder_path):
        continue

    extensions = ('.jpg')
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(extensions) and not f.startswith('.')
    ])

    for fname in files:
        label = f"img_{global_idx:04d}"
        label_to_path[label] = os.path.join(folder_path, fname)
        global_idx += 1

print(f"Построен маппинг для {len(label_to_path)} изображений")

# для каждого выброса находится ближайшее изображение, которое предположительно соответствует данной аномальной точке
# (Для каждого  выброса вычисляется евклидово расстояние до всех изображений в двумерном пространстве UMAP (косинусное для клипа, так как оно многомерное, для юмапа евклидово - двумерное))

found_outliers = []

for outlier in outliers_coords:
    distances = np.sqrt(
        (image_only_metadata["umap_x"] - outlier["x"])**2 +
        (image_only_metadata["umap_y"] - outlier["y"])**2
    )

    idx = distances.idxmin() #изображение, расположенное на минимальном расстоянии
    row = image_only_metadata.loc[idx]

    img_path = label_to_path.get(row["label"])

    found_outliers.append({
        "name": outlier["name"],
        "row": row,
        "img_path": img_path,
        "distance": distances[idx]
    }) #Сохранение информации о найденных выбросах

    print(f"\n{outlier['name']}")
    print(f"  Координаты: ({outlier['x']}, {outlier['y']})")
    print(f"  Ближайшая точка: ({row['umap_x']:.2f}, {row['umap_y']:.2f})")
    print(f"  metaphor_id: {row['metaphor_id']}, image_group: {row['image_group']}")
    print(f"  label: {row['label']}")

# Визуализация выбросов
fig, axes = plt.subplots(1, len(found_outliers), figsize=(22, 5))

for i, outlier in enumerate(found_outliers):
    ax = axes[i]

    if outlier["img_path"] and os.path.exists(outlier["img_path"]):
        img = Image.open(outlier["img_path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(
            f"{outlier['name']}\n{outlier['row']['metaphor_id']} | {outlier['row']['image_group']}\n"
            f"({outlier['row']['umap_x']:.2f}, {outlier['row']['umap_y']:.2f})",
            fontsize=10, fontweight='bold', color='darkred'
        )
    else:
        ax.text(0.5, 0.5, "Не найдено",
                ha='center', va='center', fontsize=12, color='red')
        ax.set_title(outlier['name'], fontsize=10)

    ax.axis('off')

plt.suptitle("Outliers in UMAP of Image Embeddings", fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()

"""**Евклидово расстояние**

$$
d(x, y) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$

1. Source и target разделены для каждой метафоры. То есть модель CLIP видит разницу между буквальными и метафорическими изображениями (крестики и кружочки отделены)


2. Control изображения — где-то посередине
Фиолетовые квадраты разбросаны по всему графику, не образуют четкого кластера.  Это логично — они не связаны с конкретными метафорами.

3. Разные метафоры — разные кластеры
- argument_war — левый нижний угол
- time_money — верхняя часть
- love_journey — правая часть

Это значит, что CLIP кодирует семантику метафор по-разному.

4. Target образуют более плотные кластеры

Крестики обычно компактнее, чем круги. Возможно, метафорические значения более однородны, чем буквальные.

# ***Векторная арифметика***

*Задача этого этапа:* проверить, можно ли с помощью векторной арифметики искусственно сдвинуть абстрактный target-концепт в сторону source-области соответствующей метафоры.

Для этого для каждой метафоры мы строим искусственный вектор по схеме:

$$
target.domain - abstract.anchor + source.domain
$$

Полученный вектор сравнивается с эмбеддингами всех изображений с помощью **cosine similarity**, затем мы анализируем top-10 ближайших изображений и смотрим, к каким группам они относятся: source, target или control.

Например, результирующий вектор для метафоры **"время — это деньги" = embedding("время") - embedding("абстрактное понятие") + embedding("деньги").**

То есть мы берем target-концепт “время”, убираем из него общий компонент абстрактности и пытаемся сдвинуть его в сторону source-domain “деньги”. Мы проверяем, куда указывает этот результирующий вектор?

-> Если ближайшими оказываются изображения из группы source, то у нас получилось поменять направление вектора, мы сдвинули target-концепт в сторону source-domain, метафорического источника (то есть в этом случае время в сторону денег).

-> Если ближайшими оказываются изображения из группы target, значит исходная связь с target-domain остаётся сильнее (то есть остается сильная связь со временем).

-> Если в top-10 попадает много control-изображений, то результат векторной арифметики менее интерпретируем.

*Для каждой метафоры мы будем делать свою операцию:*

1. Time is money
* time - abstract concept + money
* время - абстрактное понятие + деньги

2. Argument is war
* argument - abstract concept + war
* спор - абстрактное понятие + война

3. Love is a journey
* love - abstract emotion + journey
* любовь - абстрактная эмоция + путешествие
"""

# Функция для кодирования одного текстового промта в CLIP-эмбеддинг.
# Принимает: text — строка с текстом, который мы хотим закодировать, например: "abstract concept"
# Возвращает: нормализованный numpy-вектор размерности 512.

def encode_text_prompt(text):
    with torch.no_grad(): # отключает вычисление градиентов (мы не обучаем модель, а только используем её для получения эмбеддингов
        inputs = processor(  # processor подготавливает текст для CLIP — делает токенизацию: превращает текстовую строку в набор числовых токенов
            text=[text],
            return_tensors="pt", #  это означает: вернуть результат в формате PyTorch tensors
            padding=True, # добавляет специальные padding-токены, если тексты разной длины
            truncation=True # обрезает текст, если он слишком длинный для CLIP
        )
        # словарь с подготовленными данными
        input_ids = inputs["input_ids"].to(device) # числовые id токенов
        attention_mask = inputs["attention_mask"].to(device) # маска внимания: показывает модели, какие токены настоящие, а какие padding

        text_outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)  # model.text_model — это text encoder модели CLIP (получает числовые токены и строит для них внутренние представления)

        # text_outputs содержит несколько объектов; нам нужен pooler_output — общий вектор всего текста
        pooled_output = text_outputs.pooler_output # pooler_output — это агрегированное представление всей фразы

        # CLIP имеет две части: 1. text encoder для текстов 2. image encoder для изображений

        # Чтобы сравнивать текст и картинку через cosine similarity, их нужно привести к одной общей размерности и одному пространству

        # text_projection делает финальное преобразование текстового представления в CLIP embedding
        text_features = model.text_projection(pooled_output) # переводит выход text encoder в общее CLIP-пространство
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # считаем длину вектора и делим весь вектор на его длину (получается 1). удобно, потому что после нормализации скалярное произведение двух векторов эквивалентно cosine similarity

        return text_features.cpu().numpy()[0]

# Функция для поиска ближайших изображений
# Мы получили результирующий вектор и хотим понять, какие изображения к нему ближе.  Мы сравниваем result_vec со всеми image_embeddings и выбираем top-k изображений с самой высокой cosine similarity
# Ищем не по всему датасету, а по корректному набору кандидатов: source и target только из текущей метафоры, control оставляем для проверки шума.

def retrieve_top_images(query_vec, metaphor_id=None, top_k=10): # top_k — сколько ближайших изображений нужно вернуть, по умолчанию top_k=10
    query_vec = query_vec / np.linalg.norm(query_vec)#  query_vec — вектор-запрос, например, результат векторной арифметики; в этой строке мы его нормализуем target_vec - anchor_vec + source_vec.

    # Ставим фильтр по метафоре (чтобы исуществлять поиск внутри группы + control)
    if metaphor_id is not None:
        candidate_metadata = images_metadata[ # images_metadata — это таблица с информацией о каждом изображении
            (
                (images_metadata["metaphor_id"] == metaphor_id) &
                (images_metadata["image_group"].isin(["source", "target"]))
            )
            |
            (
                images_metadata["image_group"] == "control"
            )
        ].copy()

        candidate_indices = candidate_metadata.index.to_numpy()
        candidate_embeddings = image_embeddings[candidate_indices]

    else:
        candidate_metadata = images_metadata.copy()
        candidate_indices = candidate_metadata.index.to_numpy()
        candidate_embeddings = image_embeddings

    # Поскольку векторы нормализованы, умножаем матрицу image_embeddings на query_vec и получаем cosine similarity
    sims = candidate_embeddings @ query_vec

    top_local_idx = np.argsort(sims)[::-1][:top_k] # np.argsort(sims) возвращает индексы элементов sims в порядке возрастания, но нам нужны самые большие значения, поэтому добавляем [::-1], чтобы развернуть
    top_global_idx = candidate_indices[top_local_idx] # это индексы top-k ближайших изображений

     # Берём из images_metadata строки, соответствующие найденным индексам
    top_images = images_metadata.iloc[top_global_idx].copy()
    top_images["similarity"] = sims[top_local_idx]
    top_images["rank"] = range(1, len(top_images) + 1)  # rank показывает место изображения в выдаче: 1 — самое близкое изображение, 2 — второе по близости, 3 — третье и т.д.

    # Возвращаем таблицу с top-k изображениями, отсортированными по близости к query_vec (и только с нужными колонками)
    return top_images[
        [
            "rank",
            "image_id",
            "metaphor_id",
            "image_group",
            "image_path",
            "similarity"
        ]
    ]

# Векторная арифметика для всех метафор
# Делаем таблицу с операциями:

# - metaphor_id: к какой метафоре относится операция
# - language: язык операции
# - target: target-domain
# - anchor: абстрактный якорь
# - source: source-domain

vector_arithmetic_specs = [
    {
        "metaphor_id": "time_money",
        "language": "en",
        "target": "time",
        "anchor": "abstract concept",
        "source": "money"
    },
    {
        "metaphor_id": "time_money",
        "language": "ru",
        "target": "время",
        "anchor": "абстрактное понятие",
        "source": "деньги"
    },
    {
        "metaphor_id": "argument_war",
        "language": "en",
        "target": "argument",
        "anchor": "abstract concept",
        "source": "war"
    },
    {
        "metaphor_id": "argument_war",
        "language": "ru",
        "target": "спор",
        "anchor": "абстрактное понятие",
        "source": "война"
    },
    {
        "metaphor_id": "love_journey",
        "language": "en",
        "target": "love",
        "anchor": "abstract emotion",
        "source": "journey"
    },
    {
        "metaphor_id": "love_journey",
        "language": "ru",
        "target": "любовь",
        "anchor": "абстрактная эмоция",
        "source": "путешествие"
    }
]

# Считаем арифметику

# Создаём пустой список: в него мы будем добавлять top-10 ближайших изображений для каждой операции векторной арифметики
all_top_images = []

# Проходим циклом по всем операциям из vector_arithmetic_specs
for spec in vector_arithmetic_specs:
    target_vec = encode_text_prompt(spec["target"]) # кодируем target-domain в CLIP embedding: на выходе получаем numpy-вектор размерности 512
    anchor_vec = encode_text_prompt(spec["anchor"]) # кодируем anchor, то есть абстрактный якорь: его мы будем вычитать
    source_vec = encode_text_prompt(spec["source"]) # кодируем source-domain.

    # Собственно векторная арифметика: строим искусственный вектор по схеме: target_domain - abstract_anchor + source_domain
    result_vec = target_vec - anchor_vec + source_vec
    result_vec = result_vec / np.linalg.norm(result_vec)  # нормализуем результирующий вектор

    # Ищем top-10 среди:
    # source/target текущей метафоры + control-картинок
    top_images = retrieve_top_images(
        result_vec,
        metaphor_id=spec["metaphor_id"],
        top_k=10
    )

    top_images["query_metaphor_id"] = spec["metaphor_id"] # добавляем колонку query_metaphor_id: она показывает, для какой метафоры мы делали запрос.
    top_images["language"] = spec["language"] # добавляем колонку для языка, чтобы потом сравнить как векторны операции работают для рус/англ
    top_images["operation"] = (f'{spec["target"]} - {spec["anchor"]} + {spec["source"]}') # добавляем текстовое описание операции (чтобы показать между чем и чем происходила арифметика)

     # Добавляем полученную таблицу top-10 в общий список
    all_top_images.append(top_images)

# Объединяем все таблицы в одну большую
vector_arithmetic_results = pd.concat(all_top_images, ignore_index=True)

vector_arithmetic_results

# Сводка по результатам векторной арифметики: считаем, сколько изображений source / target / control попало в top-10 для каждой метафоры и каждого языка

vector_summary = (
    vector_arithmetic_results
    .groupby([
        "query_metaphor_id",
        "language",
        "image_group"
    ])
    .size()
    .reset_index(name="count")
)

# Широкий формат: так удобнее смотреть
vector_summary_wide = vector_summary.pivot_table(
    index=["query_metaphor_id", "language"],
    columns="image_group",
    values="count",
    fill_value=0
).reset_index()

vector_summary_wide = vector_summary_wide.astype({
    col: int for col in vector_summary_wide.columns # делаем тип int, чтобы не было .0
    if col not in ["query_metaphor_id", "language"]
})
vector_summary_wide

# ВАРИАНТ 5: ТВОИ ЦВЕТА (поменяй hex-коды на свои!)
color_map = {
    "source":  "#B0E0E6",
    "target":  "#FFB6C1",
    "control": "#66CDAA",
}


# Устанавливаем стиль
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# Создаём палитру из нашего словаря
# Порядок должен совпадать с порядком групп в данных
palette = [color_map.get(g, "#999999") for g in sorted(vector_summary["image_group"].unique())]

# Настраиваем размер шрифтов
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Создаем график
g = sns.catplot(
    data=vector_summary,
    x="query_metaphor_id",
    y="count",
    hue="image_group",
    col="language",
    kind="bar",
    height=6,
    aspect=1.3,
    palette=palette,
    sharey=False,
    edgecolor="black",
    linewidth=1.2,
    alpha=0.9,
    order=sorted(vector_summary["query_metaphor_id"].unique())  # фиксированный порядок метафор
)

# Добавляем подписи осей НА КАЖДОМ графике
for ax in g.axes.flat:
    ax.set_xlabel("Metaphor", fontsize=12, labelpad=10, fontweight='semibold')
    ax.set_ylabel("Number of images in top-10", fontsize=12, labelpad=10, fontweight='semibold')
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# Заголовки подграфиков
g.set_titles("Language: {col_name}", size=13, weight='bold', pad=12)

# Легенда
g.add_legend(
    title="Image Group",
    title_fontsize=12,
    fontsize=10,
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    edgecolor='gray'
)

# Отступы между подграфиками
g.fig.subplots_adjust(wspace=0.3)

# Общий заголовок
g.fig.suptitle(
    'Distribution of Image Groups in Top-10 Retrieval\nby Metaphor and Language',
    fontsize=15, fontweight='bold', y=1.05
)

# Сохраняем график
plt.savefig( "results/figures/vector_ariphmetics_top10.png", dpi=300, bbox_inches="tight" )

plt.show()

# Функция для вывода картинок по одной метафоре, языку и группе
def show_vector_images_by_group(
    results_df,
    metaphor_id,
    language,
    image_group,
    max_images=10
):
    subset = (
        results_df[
            (results_df["query_metaphor_id"] == metaphor_id) &
            (results_df["language"] == language) &
            (results_df["image_group"] == image_group)
        ]
        .sort_values("rank")
        .head(max_images)
    )

    if subset.empty:
        print(f"Нет изображений: {metaphor_id}, {language}, {image_group}")
        return

    n_images = len(subset)
    n_cols = min(5, n_images)
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3.5 * n_rows)
    )

    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, subset.iterrows()):
        img = Image.open(row["image_path"])
        ax.imshow(img)
        ax.axis("off")

        ax.set_title(
            f'rank={row["rank"]}\n'
            f'sim={row["similarity"]:.3f}',
            fontsize=10
        )

    # Убираем пустые ячейки, если картинок меньше, чем мест на графике
    for ax in axes[n_images:]:
        ax.axis("off")

    plt.suptitle(
        f"{metaphor_id} | {language} | {image_group}",
        fontsize=14,
        y=1.02
    )

    plt.tight_layout()
    plt.show()


# Автоматический вывод картинок:
# сначала по метафоре,
# внутри каждой метафоры по языку,
# внутри каждого языка по группам source / target / control

metaphors = vector_arithmetic_results["query_metaphor_id"].unique()
languages = vector_arithmetic_results["language"].unique()
groups = ["source", "target", "control"]

for metaphor_id in metaphors:
    print(f"Метафора: {metaphor_id}")

    for language in languages:
        print(f"\nЯзык: {language}")

        for image_group in groups:
            show_vector_images_by_group(
                vector_arithmetic_results,
                metaphor_id=metaphor_id,
                language=language,
                image_group=image_group
            )

"""**H5.** С помощью векторной арифметики можно сдвинуть абстрактный target-концепт в сторону source-области соответствующей метафоры. - **Гипотеза поддерживается.**

Почти во всех случаях в top-10 больше всего изображений из группы **source**, то есть у нас получилось поменять направление результирующего вектора в сторону source-области соответствующей метафоры за счет того, что мы вычли абстрактный концепт из target-области.

Рассмотрим более подробно каждую метафору и каждыя из языков:

**1. argument_war**

  * English

 ```text
  source = 5
  target = 5
  control = 0
  ```     

Результат смешанный: в top-10 вошло поровну source- и target-изображений. То есть вектор после операции оказался одинаково близок и к изображениям войны, и к изображениям спора. То есть метафорический сдвиг в сторону war есть, но связь с target-domain остается все еще сильной.

Control = 0 говорит о том, что случайные контрольные изображения не попали в top-10, что хорошо.
  
  * Russian

  ```text
  source = 5
  target = 1
  control = 4
  ```

Для русского результат получился чуть хуже: 5 из 10 изображений относятся к source-группе, 1 к target. Это значит, что операция сместила вектор в сторону метафоры войны.

Однако в топ-10 попало 4 контрольных изображения, поэтому результат далеко не идеальный.

**2. love_journey**

  * English

  ```text
  source = 6
  target = 3
  control = 1
  ```

Для английского большинство изображений (6) относятся к source-группе, то есть операция сместила вектор в сторону путешествия. Однако в топ-10 сохраняются 3 изображения target-группы + попало одно контрольное изображение.

  * Russian

  ```text
  source = 4
  target = 4
  control = 2
  ```

Для русского source-группа делит первое место с target (4 и 4 изображения).  При этом сюда попало 2 картинки из группы control, а значит, сдвиг в сторону source-domain есть, но при этом не такой сильный, как для английского варианта этой метафоры.

**3. time_money**

  * English

  ```text
  source = 5
  target = 4
  control = 1
  ```

Для английского результат почти сбалансированный между source и target: 5 изображений source-группы и 4 изображения target. Получается, что операция немного смещает вектор в сторону метафоры денег, но связь с target-domain остаётся почти такой же сильной. Также в то-10 попало 1 контрольное изображение.

  * Russian

  ```text
  source = 4
  target = 3
  control = 3
  ```

Для русского сдвиг в сторону метафоры денег выражен слабее, чем в английском варианте: 4 из 10 ближайших изображений относятся к source-группе, 3 относятся к target. Но в топ также попало 3 контрольных изображения, что не очень хорошо.

**Итоговый вывод**

Векторная арифметика показала хороший результат: во всех шести случаях группа **source** оказывается самой частотной или не менее частотной чем target + в топ-10 для английского языка попадает не более 1 контрольного изображения. Это значит, что векторная операция *target - abstract_anchor + source* действительно помогает сдвинуть вектор в сторону области метафоры.

Наиболее выраженный результат векторной арифметики заметен для **английского языка**, по сравнению с русским (16/30 и 13/30 изображений соответственно относятся к source-группе); кроме того, для русского значительно больше контрольных изображения попадает в топ-10, чем у английкого. Если сопоставлять метафоры, то наиболее выраженный результат получился для метафоры **love_journey**: в сумме для русского и английского варианта 10/20 изображений, попавших в топ-10 относятся к source-группе, при этом меньше всего контрольных.

# Сохранение результатов на гитхаб
"""

!find .

#!git add .

#!git commit -m "final project"

#!git push origin main

"""# Ограничения исследования и перспективы дальнейшей работы

**1. Ограниченный объём выборки**

Анализ на данном этапе включает только 3 концептуальные метафоры (но на двух языках). Кроме того, исследование проводилось на относительно небольшом количестве изображений (по 10 картинок для каждого домена, то есть по 20 картинок на метафору + 10 общих контрольных). Такой объём данных позволяет получить предварительные результаты и выявить общие тенденции, однако мы понимаем, что обобщать нужно осторожно. В дальнейшем представляется целесообразным расширить корпус изображений и увеличить количество анализируемых концептуальных метафор.

**2. Использование заранее обученной модели CLIP**

В исследовании использовалась предобученная модель CLIP, которая не адаптирована к задаче анализа концептуальных метафор и качество эмбеддингов определяется только особенностями исходного обучения модели. В перспективе можно полноценно дообучить модель на корпусах визуальных и контекстных метафор.

**3. Ограничения, касающиеся использования русского языка**

Согласно документации модели CLIP:
"the model has not been purposefully trained in or evaluated on any languages other than English": использование модели рекомендуется преимущественно для англоязычных задач, однако мы пользуемся моделью для анализа текстов как на английском, так и русском языке и понимаем, что этим как минимум частично может объясняться разница между языками и не очень высокие показатели для русского.

В дальнейшем представляется перспективным использование мультиязычных версий CLIP или специализированных мультимодальных моделей, поддерживающих русский язык, которые в данном исследовании не предоставилось возможным запустить на текущих мощностях.

**4. Субъективность при отборе метафор**

Отнесение изображений к определённым концептуальным метафорам и их компонентам (source и target domains) содержит элемент субъективности. В дальнейшем возможно привлечение нескольких экспертов и проведение оценки согласия для повышения объективности результатов.

**5. Ограничения метода UMAP**

UMAP предназначен для визуализации высокоразмерных данных и не гарантирует полного сохранения всех семантических отношений, существующих в исходном пространстве эмбеддингов, так как переводит все в двумерное пространство. В дальнейшем результаты могут быть дополнительно проверены с использованием других методов снижения размерности (например, PCA).

**6. Недостаточно подробный анализ выбросов**

Анализ выбросов проводился преимущественно визуально — это ухудшает воспроизводимость результатов. Перспективным направлением является разработка количественных критериев выявления и анализа выбросов.

# Выводы по результатам проекта

**H1.** Метафорические выражения имеют более высокое cosine similarity с изображениями, связанными с соответствующей метафорой (source и target), чем с контрольными изображениями (control).
->  Гипотеза **подтвердилась**

**H2.** Target-фразы имеют более высокое cosine similarity с изображениями target-домена, чем с изображениями source-домена.
->  Гипотеза **подтвердилась**

**H3.** Сила метафорического эффекта в пространстве эмбеддингов CLIP различается для русскоязычных и англоязычных текстов.
->  Гипотеза **подтвердилась**

**H4.** Метафорический эффект будет различаться в зависимости от конкретной метафоры.
->  Гипотеза **подтвердилась**

**H5.** С помощью векторной арифметики можно сдвинуть абстрактный target-концепт в сторону source-области соответствующей метафоры.
->  Гипотеза **подтвердилась**


Результаты показывают, что CLIP по-разному соотносит тексты и изображения в зависимости от метафоры. Некоторые метафоры дают более устойчивые и интерпретируемые связи между source- и target-доменами, чем другие, то есть одни метафоры имеют более выраженные визуальные признаки, а другие сложнее передаются через изображения. Кроме того, в некоторых случаях сдвиг от source к target помогает приблизиться к нужной метафорической области, однако результаты зависят от конкретной метафоры, языка и набора изображений.
"""
