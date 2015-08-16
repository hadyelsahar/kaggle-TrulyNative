# -*- coding: utf-8 -*-
"""

"""
#
# from lxml.html import soupparser as parser
from scrapy.selector import Selector


# Data Structure to represent document words
class Document:
    """
    class with functionality for dealing with single documents
    such as parsing html, preprocessing, character repetition removal ..etc
    """
    pageleft = 0
    status = ""

    @staticmethod
    def preprocess(text):
        text = Document.parse_html(text)
        return text

    @staticmethod
    def parse_html(text):
        """
        function to parse html as xpath and return all text inside all tags
        :param text: html raw data
        :return: all text in all tags in html concatinated
        """

        # text = Selector(text=text).xpath('//*/text()').extract()
        text = [i for i in Selector(text=text).xpath('//*/text()').extract() if len(i.strip()) > 5]
        text = " ".join(text)
        print "pages left to be parsed for %s \t%s" % (Document.status, Document.pageleft)
        Document.pageleft -= 1
        return text
