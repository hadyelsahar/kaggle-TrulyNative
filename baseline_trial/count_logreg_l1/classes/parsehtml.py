"""
instead of parsing all html files in every experiments this script is
to parse them all and load them in csv file to load them when needed.
"""

import os
from scrapy.selector import Selector
from threading import Thread
import Queue
import csv
import argparse

parser = argparse.ArgumentParser(description='script to parse all html files and convert them to raw text')
parser.add_argument('-o', '--output', help='output file that has all text', required=True)
parser.add_argument('-f', '--datapath', help='all file folder path', required=True)
args = parser.parse_args()

def read_data_files(filenames, datapath, writer, ids=None, q=None):
    """
    :param filenames: list of all file names open
    :param datapath: datapath to concatinate before each filename
    :param ids: list of all ids to parse from the list of filenames
    :param writer: csv writer to write class rows there after parsing
    :return: iterator of text containing all text in all files in order
    """
    for f in [filenames[k] for k in ids]:
        text = open(datapath+f, 'r').read()
        text = [x.strip() for x in Selector(text=text).xpath('//body//*[not(self::script) and not(self::style)]/text()').extract() if len(x.strip()) > 5]
        text = " ".join(text).encode('utf-8', 'ignore')
        writer.writerow([f, text])

        # updating global counter
        left = q.get()
        q.put(left - 1)
        if (left % 10) == 0: print "files left %s " % left

# reading all file names from data path
all_file_names = os.listdir(args.datapath)
files_count = len(all_file_names)

# creating a csv writer
fout = open(args.output, "w")
w = csv.writer(fout)
w.writerow(["filename", "file"])

# calculating ids
thread_numbers = 6
start_id = 0
end_id = 0
step = files_count/thread_numbers
thread_list = []

# counter for all number of files left to share across all threads
files_left = Queue.Queue()
files_left.put(len(all_file_names))

for i in range(thread_numbers+1):
    start_id = i * step
    end_id = start_id + step
    if end_id > files_count-1:
        end_id = files_count
    ids = range(start_id, end_id)

    t = Thread(target=read_data_files, name="thread%s" % i, args=(all_file_names, args.datapath, w, ids, files_left))
    t.start()
    print "thread%s start" % i
    thread_list.append(t)

for t in thread_list:
    t.join()

fout.close()






















