from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

batch_size = 1
test_strings = ["hello"] * batch_size
batch = tokenizer(test_strings, max_length=512, padding=True, truncation=True, return_tensors='np')
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
with open(f"models/prd/multilingual-e5-large-onnx/warmup/{batch_size}/raw_input_ids", "wb") as fh:
    fh.write(batch["input_ids"])

with open(f"models/prd/multilingual-e5-large-onnx/warmup/{batch_size}/raw_attention_mask", "wb") as fh:
    fh.write(batch["attention_mask"])


text_description = """
Встречайте iPhone 15 Pro Max - новейшее творение Apple, 
сочетающее инновационные технологии, стильный дизайн и мощные функции. 
Этот смартфон олицетворяет роскошь и высокие технологии 2024 года.

Облаченный в элегантный титановый корпус «титанового синего» цвета, 
iPhone 15 Pro Max выделяется среди конкурентов. 
Под капотом скрывается процессор A17 Pro, обеспечивающий потрясающую производительность. 
Революционная система камер с 48-мегапиксельным основным сенсором, 
12-мегапиксельной сверхширокоугольной и двумя 12-мегапиксельными телефото камерами
захватит невероятно четкие снимки. 
Технология Photonic Engine гарантирует впечатляющее качество даже в условиях низкой освещенности.

Оптический зум 5x позволяет приблизить объект без потери качества, 
а широкий диапразон 10x зума открывает безграничные творческие возможности. 
Профессиональное HDR-видео с частотой до 60 кадров/с в формате Dolby Vision 
и замедленная съемка 1080p до 240 кадров/с помогут запечатлеть самые яркие моменты. 
Режим «Киноэффект» добавит размытия заднего плана, создавая голливудский эффект в ваших роликах 4К.

Погрузитесь в мир ярких визуальных впечатлений с 6,7-дюймовым OLED-дисплеем с разрешением 2796x1290,
яркостью 2000 кд/м², контрастностью 2 000 000:1 и технологией ProMotion 120 Гц.
Всегда включенный дисплей Always-On Display позволит не упустить важную информацию.

В iPhone 15 Pro Max воплощены передовые технологии связи: 5G, Wi-Fi 6,
Bluetooth 5.3 и NFC. Экстренный вызов SOS через спутник гарантирует связь вне зон покрытия. 
Дополненная реальность раскроет новые горизонты с помощью сканера LiDAR. 
При этом смартфон обладает выдающимся временем работы без подзарядки - до 95 часов для аудио, 29 часов видео и 25 часов стриминга.

Откройте для себя iPhone 15 Pro Max от Apple – вершину инноваций среди смартфонов 2024 года. 
Насладитесь безупречным дизайном, невероятными камерами и передовыми возможностями уже сегодня!

Также доступны iPhone 13, 14 и 15, все с отличными камерами.
"""
batch_size = 256
test_strings = [text_description] * batch_size
batch = tokenizer(test_strings, max_length=512, padding=True, truncation=True, return_tensors='np')
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
with open(f"models/prd/multilingual-e5-large-onnx/warmup/{batch_size}/raw_input_ids", "wb") as fh:
    fh.write(batch["input_ids"])

with open(f"models/prd/multilingual-e5-large-onnx/warmup/{batch_size}/raw_attention_mask", "wb") as fh:
    fh.write(batch["attention_mask"])
