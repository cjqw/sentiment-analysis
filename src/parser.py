import re
import jieba

punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠~
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

def split(comment,language):
    if language == "cn":
        content = jieba.cut(comment)
        content = list(filter(lambda x: x not in punct,content))
    else:
        content = re.split(r'[\.,;!?\'\s]+',comment)
    return content
