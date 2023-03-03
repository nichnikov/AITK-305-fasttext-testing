import os
from src.config import PROJECT_ROOT_DIR
from navec import Navec
from razdel import sentenize, tokenize
from slovnet import (NER,
                     Syntax,
                     Morph)
from ipymarkup import show_span_ascii_markup as show_markup

text = "Если нет выплат, то надо ли сдавать персонифицированный отчет."
# text = 'Европейский союз добавил в санкционный список девять политических деятелей из самопровозглашенных республик Донбасса — Донецкой народной республики (ДНР) и Луганской народной республики (ЛНР) — в связи с прошедшими там выборами. Об этом говорится в документе, опубликованном в официальном журнале Евросоюза. В новом списке фигурирует Леонид Пасечник, который по итогам выборов стал главой ЛНР. Помимо него там присутствуют Владимир Бидевка и Денис Мирошниченко, председатели законодательных органов ДНР и ЛНР, а также Ольга Позднякова и Елена Кравченко, председатели ЦИК обеих республик. Выборы прошли в непризнанных республиках Донбасса 11 ноября. На них удержали лидерство действующие руководители и партии — Денис Пушилин и «Донецкая республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР. Президент Франции Эмманюэль Макрон и канцлер ФРГ Ангела Меркель после встречи с украинским лидером Петром Порошенко осудили проведение выборов, заявив, что они нелегитимны и «подрывают территориальную целостность и суверенитет Украины». Позже к осуждению присоединились США с обещаниями новых санкций для России.'

navec = Navec.load(os.path.join(PROJECT_ROOT_DIR, "data", 'navec_news_v1_1B_250K_300d_100q.tar'))
ner = NER.load(os.path.join(PROJECT_ROOT_DIR, "data", "slovnet_ner_news_v1.tar"))
ner.navec(navec)
markup = ner(text)
print(ner(text))
show_markup(markup.text, markup.spans)


chunk = []
for sent in sentenize(text):
    tokens = [_.text for _ in tokenize(sent.text)]
    chunk.append(tokens)

print("chunk:\n", chunk)
print("chunk:\n", chunk[:1])
navec = Navec.load(os.path.join(PROJECT_ROOT_DIR, "data", 'navec_news_v1_1B_250K_300d_100q.tar'))
morph = Morph.load(os.path.join(PROJECT_ROOT_DIR, "data", 'slovnet_morph_news_v1.tar'), batch_size=4)
morph.navec(navec)
markup = next(morph.map(chunk))
for token in markup.tokens:
    print(f'{token.text:>20} {token.tag}')

syntax = Syntax.load(os.path.join(PROJECT_ROOT_DIR, "data", 'slovnet_syntax_news_v1.tar'))
syntax.navec(navec)
markup = next(syntax.map(chunk))

# Convert CoNLL-style format to source, target indices
words, deps = [], []
for token in markup.tokens:
    words.append(token.text)
    source = int(token.head_id) - 1
    target = int(token.id) - 1
    if source > 0 and source != target:  # skip root, loops
        deps.append([source, target, token.rel])

print(words, deps)
deps2 = []
for x in deps:
    if x[0] < x[1]:
        deps2.append([x[0], x[1], x[2]])
    else:
        deps2.append([x[1], x[0], x[2]])

print(deps2)
show_markup(words, deps2)