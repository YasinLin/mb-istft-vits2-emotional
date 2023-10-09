class VITS_BERT:
    def __init__(self, bert_path, device):
        self.prosody = TTSProsody(bert_path, device)

    def chinese_to_bert(self, text):
        # 将标准中文文本符号替换成 bert 符号库中的单符号,以保证bert的效果.
        text = text.replace("——", "...")\
            .replace("—", "...")\
            .replace("……", "...")\
            .replace("…", "...")\
            .replace('“', '"')\
            .replace('”', '"')
        text = number_to_chinese(text)
        text = text.replace('、', '，').replace('；', '，').replace('：', '，')
        words = jieba.lcut(text, cut_all=False)
        word2num = []
        word2num.append(1)
        for word in words:
            bopomofos = lazy_pinyin(word, BOPOMOFO)
            if not re.search('[\u4e00-\u9fff]', word):
                match = False
                for regex, replacement in _latin_to_bopomofo:
                    if regex.search(word):
                        word = re.sub(regex, replacement, word)
                        word2num.append(len(word))
                        match = True
                if not match:
                    for c in word:
                        word2num.append(1)
                
                continue
            
            for i in range(len(bopomofos)):
                bopomofos[i] = re.sub(r'([\u3105-\u3129])$', r'\1ˉ', bopomofos[i])
                word2num.append(len(bopomofos[i]))

            
        word2num.append(1)
        text = text.replace('\n', '[UNK]')\
            .replace(' ', '[UNK]')
        text = f'[PAD]{text}[PAD]'
        char_embeds = self.prosody.get_char_embeds(text)
        input = char_embeds
        try:
            char_embeds = self.prosody.expand_for_phone(char_embeds,word2num)
        finally:
            # pass
            print(text, word2num,sum(word2num), input.size(0))
        return char_embeds


vits_bert = None

def get_vits_bert():
    global vits_bert
    if not vits_bert:
        vits_bert = VITS_BERT("models/bert-prosody/", "cuda")
    return vits_bert
