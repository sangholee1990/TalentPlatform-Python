import os
import pandas as pd
import csv

def split_csv_by_cutoff():
    files = [
        'B_8000765_W24009_MAIN.CSV',
        'B_8000765_W25004_MAIN.CSV',
        'B_8000765_W23008_MAIN.CSV',
        'B_8000765_W24001_MAIN.CSV',
        'B_8000765_W25007_MAIN.CSV',
        'B_8000765_W25003_MAIN.CSV'
    ]
    cutoffs = [1, 3, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    input_dir = 'data'
    output_dir = 'data_split_by_cutoff'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[{output_dir}] 폴더를 생성하고 데이터를 분리합니다...")
    
    for file in files:
        filepath = os.path.join(input_dir, file)
        if not os.path.exists(filepath):
            print(f"파일을 찾을 수 없습니다: {filepath}")
            continue
            
        # 첫 3줄(헤더 및 단위/설명 부분)을 원본 그대로 유지하기 위해 직접 읽기
        with open(filepath, 'r', encoding='cp949') as f:
            lines = f.readlines()
            
        if len(lines) <= 3:
            continue
            
        # CSV reader를 사용하여 Age 컬럼 인덱스 찾기
        header_row = next(csv.reader([lines[0]]))
        try:
            age_idx = header_row.index('Age')
        except ValueError:
            print(f"'Age' 컬럼을 찾을 수 없습니다: {file}")
            continue
            
        for cutoff in cutoffs:
            out_lines = lines[:3] # 헤더 3줄 포함
            
            data_added = False
            for line in lines[3:]:
                # csv.reader로 파싱하여 값 확인
                row = next(csv.reader([line]))
                if len(row) > age_idx:
                    age_str = row[age_idx].strip()
                    if not age_str:
                        continue
                    try:
                        age_val = float(age_str)
                        if age_val <= cutoff:
                            out_lines.append(line)
                            data_added = True
                        else:
                            break # 시간이 초과되면 중단
                    except ValueError:
                        pass
                        
            if data_added:
                out_name = f"{file.replace('.CSV', '')}_cutoff_{cutoff}h.CSV"
                out_path = os.path.join(output_dir, out_name)
                with open(out_path, 'w', encoding='cp949') as f:
                    f.writelines(out_lines)
        print(f"완료: {file}")
        
if __name__ == "__main__":
    split_csv_by_cutoff()
