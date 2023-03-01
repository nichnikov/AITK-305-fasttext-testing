import os
import fasttext

# ft = download_model('ru')
model = fasttext.load_model(os.path.join("models", "cc.ru.300.bin"))
print(model.get_nearest_neighbors('ндфл', k=50))
print(model.get_nearest_neighbors('ндс', k=50))
print(model.get_nearest_neighbors('земельному', k=50))
print(model.get_nearest_neighbors('транспортному', k=50))
# print(model.get_nearest_neighbors('транспортному налогу', k=50))
