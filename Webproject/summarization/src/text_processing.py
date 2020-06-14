import spacy
from others.logging import logger

def text_refine(inpath, outpath, finalpath):
    nlp = spacy.load('en')
    f1 = open(outpath, "r", encoding='utf-8')
    f2 = open(inpath, "r", encoding='utf-8')
    out_texts = f1.readlines()
    in_texts = f2.readlines()
    f1.close()
    f2.close()
    f = open(finalpath, 'w', encoding='utf-8')
    for idx in range(len(out_texts)):
        out_text = out_texts[idx].replace('\n', '.').replace('<q>', '. ')
        in_text = in_texts[idx].strip('/n')
        out_doc = nlp(out_text)
        in_doc = nlp(in_text)
        in_dic = {}
        for sent in in_doc.sents:
            #print("SENT: %s" % sent)
            firstWord = False
            meetQuo = False
            for idx, token in enumerate(sent):
                if token.text == '\"':
                    meetQuo = True
                if token.text.isalpha():
                    if not firstWord:
                        firstWord = True
                        continue
                    if meetQuo:
                        meetQuo = False
                        continue
                    if not token.text.islower():
                        #print(token.text.lower())
                        #print("SENT: %s" % sent)
                        in_dic[token.text.lower()] = token.text 
        #print(in_dic)
        out_text = ""
        for sent in out_doc.sents:
            if out_text != "":
                out_text = out_text + " "
            out_text = out_text + sent.text.capitalize()
        out_doc = nlp(out_text)
        lst = [tok.text_with_ws for tok in out_doc]
        for idx in range(len(lst)):
            tok_ws = lst[idx]
            tok = tok_ws.strip()
            pos = tok_ws.find(tok)
            if tok in in_dic:
                lst[idx] = tok_ws[: pos] + in_dic[tok] + tok_ws[pos+len(tok): ]
        out_text = "".join([tok for tok in lst])
        print(out_text)
        f.write(out_text + '\n')
    f.close()

    logger.info('Refined output texts are saved at %s' % finalpath)


if __name__ == '__main__':
    text_refine("raw_data/example.raw_src", "results/cnndm.-1.candidate", "results/final.opt")