import logging
import os
import sys
from datetime import datetime
from config import settings as config

def setup_logger():
    """프로젝트 전반에 사용될 로거 설정.

    로그는 INFO 레벨 이상으로 설정되며, 콘솔과 로그 파일에 모두 출력.
    로그 파일은 `output/backtest_log_YYYY-MM-DD_HHMMSS.log` 형식으로 저장.
    """
    # 루트 로거의 기존 핸들러 제거 (중복 출력 방지)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_filename = f"backtest_log_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
    log_filepath = os.path.join(config.OUTPUT_DIR, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s]: %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger()
    logger.info("로거 설정 완료")
    logger.info(f"로그 파일 생성 위치: {log_filepath}")
    return logger

# 모듈 로드 시 로거 설정
logger = setup_logger()