# -*- coding: utf-8 -*-
import argparse
import os
from datetime import datetime
import warnings
import yaml
import pygrib

def main () :
	try:
		inFile=""
		gribDatas=pygrib.open(inFile)
		for gribData in gribDatas :
			print(gribData)
		

	except KeyError as e:
		common.logger.error("check the argments or data type or path" + e)      


if __name__ =='__main__':
    main()
