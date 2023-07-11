import os
import json
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from starlette.responses import StreamingResponse
import traceback
import json

if __name__ == '__main__':

    globalVar = {}
    parser = argparse.ArgumentParser()
    
    for i, argv in enumerate(sys.argv[1:]):
        if not argv.__contains__('--'): continue
        parser.add_argument(argv)
    
    inParInfo = vars(parser.parse_args())
    #print(inParInfo)
    
    for key, val in inParInfo.items():
        if val is None: continue
        # 전역 변수에 할당
        globalVar[key] = val
    
    
    #print(f'[CHECK] globalVar : {globalVar}')
    
    result = None
    
    try:
#        print('[SRTART] {}'.format(globalVar['cmd']))
    
        x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
        cos, sin = np.cos(x), np.sin(x)
        
        # 출력된 각 줄의 길이를 그래프로 그립니다.
        plt.figure(figsize=(10, 6))
        plt.plot(x, cos, color="blue", linewidth=2.5, linestyle="-", label="cosine")
        plt.plot(x, sin, color="red",  linewidth=2.5, linestyle="-", label="sine")
        
        plt.legend(loc='upper left', frameon=False)
        
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
        
        plt.yticks([-1, 0, +1],
                   [r'$-1$', r'$0$', r'$+1$'])
        
        # 그래프를 이미지로 저장하고 그것을 바이트로 변환합니다.
        #buf = BytesIO()
        
        #saveImg = '{}/{}/{}.png'.format(globalVar['output'], datetime.now().strftime("%Y%m%d%H%M%S"))
        #saveImg = '{}/{}/{}.png'.format(globalVar['output'], datetime.now().strftime("%Y%m%d%H%M%S"))
        saveImg = globalVar['output']
        plt.savefig(saveImg, dpi=600, bbox_inches='tight')
        #plt.savefig(buf, format='png')
        #buf.seek(0)
        
        result = {
            'msg': 'succ'
            , 'saveImg': saveImg
            , 'isExist': os.path.exists(saveImg) 
        }
        
        print(json.dumps(result))
    
    except Exception as e:
         print(traceback.format_exc())
