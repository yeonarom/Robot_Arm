# Robot_Arm
로봇 팔이 물체를 잡아 바구니에 담는 Task

## agents/models
학습 모델 파일을 위한 디렉토리
- ball-v0, ball-v1 : 각기 다른 3가지 색상의 물체를 동일한 색상의 바구니에 담는 Task
- single_vall-v1 : 하나의 물체를 바구니에 담는 Task

## environments/simple_env/envs
학습 환경 세팅 및 로봇 학습을 위한 디렉토리
- single_ball.py : 하나의 물체와 하나의 바구니를 로봇 팔 앞에 세팅
- triple_ball.py : 3가지 색상의 물체와 물체의 색상과 일치하는 3가지 색상의 바구니를 로봇 팔 앞에 세팅

## model_test.py
학습된 모델을 테스트하기 위한 코드
