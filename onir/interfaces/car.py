from onir import util

try:
    import trec_car.read_data
except ImportError:
    print('failed to import trec_car')


@util.path_to_stream(mode='rb')
def iter_paras(cbor_file):
    paras = trec_car.read_data.iter_paragraphs(cbor_file)
    for p in paras:
        yield (p.para_id, p.get_text())


@util.path_to_stream(mode='rb')
def iter_queries(cbor_file):
    for page in trec_car.read_data.iter_outlines(cbor_file):
        for heads in page.flat_headings_list():
            qid = '/'.join([page.page_id] + [h.headingId for h in heads])
            text = [page.page_name] + [h.heading for h in heads]
            yield qid, text
