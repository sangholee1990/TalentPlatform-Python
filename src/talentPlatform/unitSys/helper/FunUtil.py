# -*- coding: utf-8 -*-

import numpy as np

# 1. 구글드라이브에 python 스크립트 파일을 만들고, 아래의 함수 정의
# y=f(A,B) = 5*A+3*B + 10 -> 여기서 A와 B는 numpy로 정의된 행렬에 해당

def matCalc(A, B):

    # B 행렬 (2x1 > 2x3 추가)
    initB = B
    B = np.append(B, initB, axis=1)
    B = np.append(B, initB, axis=1)

    npA = np.array(A)
    npB = np.array(B)

    print("[CHECK] npA: {}".format(npA))
    print("[CHECK] npB: {}".format(npB))

    val = (5 * npA) + (3 * npB) + 10

    return val