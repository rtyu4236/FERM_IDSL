import os
import pandas as pd
import glob

# 스크립트가 실행되는 위치를 기준으로 프로젝트 루트 디렉터리를 설정합니다.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def _optimize_df_memory(df):
    """메모리 절약을 위해 숫자형 컬럼의 데이터 타입을 다운캐스팅합니다."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    print("  - 데이터 타입 메모리 최적화 완료.")
    return df

def convert_csv_to_parquet():
    """
    data 디렉터리 내의 모든 CSV 파일을 찾아 메모리 최적화된 Parquet 포맷으로 변환합니다.
    'date' 컬럼은 datetime 타입으로 변환됩니다.
    """
    if not os.path.exists(DATA_DIR):
        print(f"오류: 'data' 디렉터리를 찾을 수 없습니다: {DATA_DIR}")
        return

    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    if not csv_files:
        print(f"'{DATA_DIR}' 디렉터리에서 CSV 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 CSV 파일을 Parquet으로 변환합니다...")

    for csv_path in csv_files:
        file_name = os.path.basename(csv_path)
        parquet_path = os.path.splitext(csv_path)[0] + '.parquet'
        
        print(f"\n[ 1/2 ] 처리 중: {file_name}")

        try:
            # CSV 파일 읽기
            df = pd.read_csv(csv_path, low_memory=False)
            df.columns = [col.lower() for col in df.columns]

            # 날짜 컬럼을 datetime 타입으로 변환 (중요)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                print(f"  - 'date' 컬럼을 datetime 타입으로 변환 완료.")
            else:
                print("  - 'date' 컬럼이 없어 날짜 변환을 건너뜁니다.")

            # 메모리 최적화
            df_optimized = _optimize_df_memory(df)

            # Parquet 파일로 저장
            df_optimized.to_parquet(parquet_path, index=False)
            print(f"[ 2/2 ] 변환 완료: {os.path.basename(parquet_path)}")

        except FileNotFoundError:
            print(f"  - 오류: 파일을 찾을 수 없습니다: {csv_path}")
        except Exception as e:
            print(f"  - {file_name} 처리 중 오류 발생: {e}")

    print("\n모든 CSV 파일의 Parquet 변환이 완료되었습니다.")

if __name__ == "__main__":
    convert_csv_to_parquet()