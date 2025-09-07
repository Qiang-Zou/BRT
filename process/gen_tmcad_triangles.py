import logging
from solid_to_triangles2 import process_main
import argparse
import os.path
if not os.path.exists('logs'):
    os.makedirs('logs',exist_ok=True)

logging.basicConfig(filename='logs/solid_to_triangles', filemode='w',
                    format=' %(asctime)s :: %(levelname)-8s :: %(message)s',level=logging.INFO)

paser=argparse.ArgumentParser('Convert each face of solid models into triangular beziers')
paser.add_argument('data_path',type=str)
paser.add_argument('output_path',type=str)
args=paser.parse_args()
process_main(args.data_path,args.output_path,method=8,dataset='tmcad',target='triangles',process_num=30)