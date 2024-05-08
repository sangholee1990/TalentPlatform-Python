#!/bin/bash

# 사용 환경에 맞게 grep 대상은 변경 필요
stop_target=`ps -ef |grep python |grep "rcmdapt_ai_server.py" |awk '{print$2}'`

for each_target in $stop_target; do
    sudo kill -9 $each_target
done

exit
