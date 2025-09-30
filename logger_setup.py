import logging
import os
import sys
from datetime import datetime
import config

def setup_logger():
    """
    프로젝트 전반에 사용될 로거를 설정.

    - 로그 레벨: INFO
    - 출력: 콘솔(stdout)과 파일(output/backtest_log_YYYY-MM-DD_HHMMSS.log)에 동시 출력
    - 포맷: [시간] [로그레벨]: 메시지
    """
    # 루트 로거의 기존 핸들러 제거 (중복 출력 방지)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 로그 파일 경로 설정
    log_filename = f"backtest_log_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
    log_filepath = os.path.join(config.OUTPUT_DIR, log_filename)

    # 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 전역 로거 가져오기
    logger = logging.getLogger()
    logger.info("Logger setup complete.")
    logger.info(f"Log file created at: {log_filepath}")
    return logger

# 모듈 로드 시 로거 설정
logger = setup_logger()
