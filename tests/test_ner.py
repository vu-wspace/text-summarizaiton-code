from collections import Counter

from langchain_core.documents import Document

from text_summarization.ner import merge_aliases, ner_text


def test_ner_text_includes_title_and_abstract():
    d = Document(page_content='Theo lời khai của Huy tại phiên toà .',
                 metadata={'title': 'Bản án lừa đảo', 'abstract': 'TAND tuyên phạt Hồ Xuân Huy 12 năm tù .'})
    assert ner_text(d) == 'Bản án lừa đảo . TAND tuyên phạt Hồ Xuân Huy 12 năm tù . Theo lời khai của Huy tại phiên toà .'


def test_ner_text_skips_missing_abstract():
    d = Document(page_content='Thân bài .', metadata={'title': 'Tiêu đề', 'abstract': ''})
    assert ner_text(d) == 'Tiêu đề . Thân bài .'


def test_short_name_merged_into_full_name_from_abstract():
    ents = Counter({('Hồ Xuân Huy', 'PERSON'): 1, ('Huy', 'PERSON'): 5, ('Huy', 'LOCATION'): 1})
    assert merge_aliases(ents) == {('Hồ Xuân Huy', 'PERSON'): 6, ('Huy', 'LOCATION'): 1}
