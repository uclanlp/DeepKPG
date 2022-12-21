import csv
import sys
import os
import gzip
import json

# In the corpus tsv, each docid occurs at offset docoffset[docid]
docoffset = {}
with gzip.open("msmarco-docs-lookup.tsv.gz", 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
        docoffset[docid] = int(offset)


def getcontent(docid, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), "Looking for {}, found {}".format(docid, line)
    return line.rstrip().split("\t")


def generate(query_file, qrels_file, outfile, orcas_file=None):
    """Generates triples comprising:
    - Query: The current topicid and query string
    - Pos: One of the positively-judged documents for this query
    Since we have the URL, title and body of each document, this gives us six columns in total:
    topicid, query, posdocid, posurl, postitle, posbody
    outfile: The filename where the triples are written
    """

    # For each docid, the list of relevant topics - doc_to_topicid[docid]
    doc_to_topicid = {}
    with gzip.open(qrels_file, 'rt', encoding='utf8') as f:
        for line in f:
            [topicid, _, docid, rel] = line.split()
            assert rel == "1"
            if docid in doc_to_topicid:
                doc_to_topicid[docid].append(topicid)
            else:
                doc_to_topicid[docid] = [topicid]

    topic_to_querystring = {}
    with gzip.open(query_file, 'rt', encoding='utf8') as qf:
        tsvreader = csv.reader(qf, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            topic_to_querystring[topicid] = querystring_of_topicid

    if orcas_file:
        with gzip.open(orcas_file, 'rt', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [qid, query, docid, _] in tsvreader:
                if docid in doc_to_topicid:
                    doc_to_topicid[docid].append(qid)
                else:
                    doc_to_topicid[docid] = [qid]
                if qid not in topic_to_querystring:
                    topic_to_querystring[qid] = query

    csv.field_size_limit(sys.maxsize)
    with open("msmarco-docs.tsv", encoding="utf8") as fin, \
            open(outfile, 'w', encoding="utf8") as out:
        for docid in doc_to_topicid.keys():
            doc = getcontent(docid, fin)
            if len(doc) < 4:
                continue
            if doc[0] in doc_to_topicid:
                tids = doc_to_topicid[doc[0]]
                queries = [topic_to_querystring[t] for t in tids
                           if t in topic_to_querystring]
                abstract = doc[3]
                # we consider a maximum length of 1000
                abstract = ' '.join(abstract.split()[:1000])
                if queries:
                    ex = {
                        'id': docid,
                        'url': doc[1],
                        'title': doc[2],
                        'abstract': abstract,
                        'keywords': ';'.join(queries)
                    }
                    out.write(json.dumps(ex) + "\n")


if __name__ == '__main__':
    if not os.path.exists('train.json'):
        generate("msmarco-doctrain-queries.tsv.gz",
                 "msmarco-doctrain-qrels.tsv.gz",
                 "train.json", orcas_file='orcas.tsv.gz')
    if not os.path.exists('valid.json'):
        generate("msmarco-docdev-queries.tsv.gz",
                 "msmarco-docdev-qrels.tsv.gz",
                 "valid.json")
