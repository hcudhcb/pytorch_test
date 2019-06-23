#!/usr/bin/python
# -*- coding:utf-8 -*-
# Author:Yuanyuan Zhang

import re

class data_process(object):
    def __init__(self):
        # Hashtags
        self.hash_regex = re.compile(r"#(\w+)")  # 匹配话题符号#及后面的内容
        # Handels
        self.hndl_regex = re.compile(r"@(\w+)")  # 匹配@及其内容
        # URLs
        self.url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")  # 匹配链接
        # Spliting by word boundaries
        self.word_bound_regex = re.compile(r"\W+")
        # Repeating words like hurrrryyyyyy
        self.rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);  # 匹配重复单字符 .匹配任意除\n外的单字符
        # Emoticons
        self.emoticons = \
            [('__EMOT_SMILEY', [':-)', ':)', '(:', '(-:', ]), \
             ('__EMOT_LAUGH', [':-D', ':D', 'X-D', 'XD', 'xD', ]), \
             ('__EMOT_LOVE', ['<3', ':\*', ]), \
             ('__EMOT_WINK', [';-)', ';)', ';-D', ';D', '(;', '(-;', ]), \
             ('__EMOT_FROWN', [':-(', ':(', '(:', '(-:', ]), \
             ('__EMOT_CRY', [':,(', ':\'(', ':"(', ':((']), \
             ]

        # Punctuations
        self.punctuations = \
            [  # ('',		['.', ] )	,\
                # ('',		[',', ] )	,\
                # ('',		['\'', '\"', ] )	,\
                ('__PUNC_EXCL', ['!', '¡', ]), \
                ('__PUNC_QUES', ['?', '¿', ]), \
                ('__PUNC_ELLP', ['...', '…', ]), \
                # FIXME : MORE? http://en.wikipedia.org/wiki/Punctuation
            ]
        self.emoticons_regex = [(repl, re.compile(self.regex_union(self.escape_paren(regx)))) \
                                for (repl, regx) in self.emoticons]


    def hash_repl(self,match):
        return '__HASH_' + match.group(1).upper()



    def hndl_repl(self,match):
        return '__HNDL'  # _'+match.group(1).upper()

    def rpt_repl(self,match):
        return match.group(1) + match.group(1)

    # Printing functions for info
    def print_config(self,cfg):
        for (x, arr) in cfg:
            print(x, '\t', )
            for a in arr:
                print(a, '\t', )
            print('')

    def print_emoticons(self):
        self.print_config(self.emoticons)

    def print_punctuations(self):
        self.print_config(self.punctuations)

    # For emoticon regexes
    def escape_paren(self,arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def regex_union(self,arr):
        return '(' + '|'.join(arr) + ')'



    # For punctuation replacement
    def punctuations_repl(self,match):
        text = match.group(0)
        repl = []
        for (key, parr) in self.punctuations:
            for punc in parr:
                if punc in text:
                    repl.append(key)
        if (len(repl) > 0):
            return ' ' + ' '.join(repl) + ' '
        else:
            return ' '

    def processHashtags(self, text, subject='', query=[]):
        return re.sub( self.hash_regex, self.hash_repl, text )

    def processHandles( self,text, subject='', query=[]):
        return re.sub( self.hndl_regex, self.hndl_repl, text )

    def processUrls(self,text, subject='', query=[]):
        return re.sub( self.url_regex, ' __URL ', text )

    def processEmoticons(self,text, subject='', query=[]):
        for (repl, regx) in self.emoticons_regex :
            text = re.sub(regx, '  ' +self.repl +' ', text)
        return text

    def processPunctuations(self,text, subject='', query=[]):
        return re.sub( self.word_bound_regex , self.punctuations_repl, text )

    def processRepeatings(self,text, subject='', query=[]):
        return re.sub( self.rpt_regex, self.rpt_repl, text )

    def processQueryTerm(self,text, subject='', query=[]):
        query_regex = "|".join([ re.escape(q) for q in query])
        return re.sub( query_regex, '__QUER', text, flags=re.IGNORECASE )

    def countHandles(self,text):
        return len( re.findall( self.hndl_regex, text) )
    def countHashtags(self,text):
        return len( re.findall( self.hash_regex, text) )
    def countUrls(self,text):
        return len( re.findall( self.url_regex, text) )
    def countEmoticons(self,text):
        count = 0
        for (repl, regx) in self.emoticons_regex :
            count += len( re.findall( regx, text) )
        return count

    # FIXME: preprocessing.preprocess()! wtf! will need to move.
    # FIXME: use process functions inside
    def processAll( self,text, subject='', query=[]):
        if(len(query)>0):
            query_regex = "|".join([ re.escae(q) for q in query])
            text = re.sub( self.query_rgex, '__QUER', text, flags=re.IGNORECASE )

        text = re.sub( self.hash_regex, self.hash_repl, text )
        text = re.sub( self.hndl_regex, self.hndl_repl, text )
        text = re.sub( self.url_regex, ' __URL ', text )

        for (repl, regex) in self.emoticons_regex :
            text = re.sub(regex, ' '+repl+'  ' , text)

        text = text.replace('\' ','')
        # FIXME: Jugad

        text = re.sub( self.word_bound_regex , self.punctuations_repl, text)
        text = re.sub( self.rpt_regex, self.rpt_repl, text)
        return text