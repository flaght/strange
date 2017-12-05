# coding: utf-8
from javis_data import JavisData
import argparse


FLAGS = None
class StrangeEngine:
    def __init__(self):
        self.__javis_data= JavisData()

    def create_javis(self, uid, token):
        self.__javis_data.create_javis(uid, token)

    def build_dominat(self, path, filename):
        self.__javis_data.calut_dominant(path,filename)

    def build_dominat_bar(self, filename, out_dir):
        self.__javis_data.build_domiant_bar(filename, out_dir)

    def check_data(self, dir):
        self.__javis_data.check_data_dir(dir)

    def check_data_file(self, filename):
        self.__javis_data.check_data_file(filename)

if __name__ == '__main__':
    parsed = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int)
    parser.add_argument('--check_dir', type=str)
    parser.add_argument('--check_file', type=str)
    parser.add_argument('--dominat_file', type=str)
    parser.add_argument('--out_dir', type=str)
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.type == None:
        print "4. check data"
    else:
        clt = JavisData()
        if FLAGS.type == 3:
            clt.build_dominat_bar(FLAGS.dominat_file, FLAGS.out_dir)
        elif FLAGS.type == 4:
            clt.check_data_dir(FLAGS.check_dir)
        elif FLAGS.type == 5:
            clt.check_data_file(FLAGS.check_file)
    '''
    uid = 142
    token = 'adc28ac69625652b46d5c00b'
    clt =  JavisData()
    
    clt.build_dominat_bar('./data/temp/dominant.csv', './data/out_dir')
    key_dict = [
                #{'symbol':'ag1601.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-1-15 15:00:00'}
                #{'symbol':'ag1602.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-2-5 15:00:00'},
                #{'symbol':'ag1603.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-3-15 15:00:00'}, 
                #{'symbol':'ag1604.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-4-15 15:00:00'},
                #{'symbol':'ag1605.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-5-16 15:00:00'},
                #{'symbol':'ag1606.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-6-14 15:00:00'}, 
                #{'symbol':'ag1607.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-7-15 15:00:00'},
                #{'symbol':'ag1608.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-8-15 15:00:00'},
                #{'symbol':'ag1609.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-9-19 15:00:00'},
                #{'symbol':'ag1610.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-10-17 15:00:00'},
                #{'symbol':'ag1611.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-11-15 15:00:00'},
                #{'symbol':'ag1612.SHFE','start_time':'2016-1-4 9:00:00','end_time':'2016-12-15 15:00:00'},
                #{'symbol':'ag1701.SHFE','start_time':'2016-1-18 9:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1702.SHFE','start_time':'2016-2-16 9:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1703.SHFE','start_time':'2016-3-16 9:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1704.SHFE','start_time':'2016-4-18 9:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1705.SHFE','start_time':'2016-5-17 9:00:00','end_time':'2016-12-30 15:00:00'}, 
                #{'symbol':'ag1706.SHFE','start_time':'2016-6-17 9:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1707.SHFE','start_time':'2016-7-18 9:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1708.SHFE','start_time':'2016-8-16 9:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1709.SHFE','start_time':'2016-9-20 9:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1710.SHFE','start_time':'2016-10-18 9:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1711.SHFE','start_time':'2016-11-16 09:00:00','end_time':'2016-12-30 15:00:00'},
                #{'symbol':'ag1712.SHFE','start_time':'2016-12-16 09:00:00','end_time':'2016-12-30 15:00:00'}
            ]
    
  #  for key in key_dict:
  #      print key['symbol'], key['start_time'], key['end_time']
  #      print clt.send_data_pd(key['symbol'], key['start_time'], key['end_time'])
  #      print "==>"
  '''
