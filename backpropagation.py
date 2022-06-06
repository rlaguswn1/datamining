# 오차 역전파법 코딩을 해봅시다
'''
각 층 노드 2
은닉 계층 1
출력 계층 1

교안 바탕으로 해야할 것들 순서대로 정리
초기값 할당 > Forward Pass > Backward Pass > Forward Pass

--------------------------------
상세내용

1. 초기값 할당
 - 각 노드, 가중치, bias에 대한 초기값을 할당해야함
 - 랜덤이라 하셨지만... 교안의 값과 비슷한지 확인하기 위해 예제와 동일한 초기값 이용

2. Forward Pass
 - 실제값 구하기
  1) h1의 입력값 계산 (w1*i1+w2*i2+b1*1)
  2) 해당 노드의 출력값 계산 (시그모이드)
  3) 2)의 출력값을 h1노드의 값으로 생각하고 해당 위치의 가중치 및 bias 이용하여 1)~2) 반복

 - 에러 구하기
  1) sum(((1/2)*(목표값-실제값)^2)
   ㄴ 목표값: o1, o2의 초기값
  2) o1, o2에 대해 계산한 후 합치기
  
3. Backward Pass - Output layer
 - 다소 복잡해보인다.. 나름 맥락으로 이해를 해보자면
  1) 이짓거리를 왜하냐?
   - 에러(Et)가 얼마나 났나 보고 가중치(wn)들을 조정하려고
     >> wn에 대한 Et의 변화율: d(Et)/d(wn)
  2) 근데 저 식을 한번에 계산할 수 없다. 중간에 껴있는 애들이 있어서 그런가 몰라도... 어쨌든 chain rule 이용
   - o1이 계산된 과정을 w5의 시선에서 보자면:
     w5 > net_o1 > out_o1
     >> 쟤네는 서로서로 의존하는 관계
   - 그래서 결국:
     d(Et)/d(w5) = d(Et)/d(out_o1)*d(out_o1)/d(net_o1)*d(net_o1)/d(w5)
    - 각 항의 미분되는 변수에 관련한 식을 편미분하는 식으로 계산
  3) 도출된 값을 바탕으로 w5 업데이트
   - w5-학습률*(d(Et)/d(w5))
  4) 1)~3)을 w6, w7, w8에 대해서도 진행 

4. Backward Pass - Hidden layer
 - 여기서 얻어야할 것은 d(Et)/d(wn)

 1) 3.과 같은 이유로
   d(Et)/d(w1)=d(Et)/d(out_h1)*d(out_h1)/d(net_h1)*d(net_h1)/d(w1)

 2) 근데 여기는.. 출력계층 노드 2개에 연결되어있고, 오류 발생에 기여한 것이 각각 있기 때문에 두 노드의 에러를 모두 고려해야함
   d(Et)/d(out_h1)=d(Eo1)/d(out_h1)+d(Eo2)/d(out_h1)

  - 여기서 chain rule
   d(Eo1)/d(out_h1) = d(Eo1)/d(net_o1)*d(net_o1)/d(out_h1)
   d(Eo1)/d(net_h1) = d(Eo1)/d(out_o1)*d(out_o1)/d(net_h1)

 3) Eo2도 똑같이 계산

 4) 도출된 값으로 w1 업데이트:
  - w1-학습률*(d(Et)/d(w1))
 
 5) w2, w3, w4에 대해서도 똑같이

5. Forward Pass 한 번 더 돌려서 오차 확인

-------------------------------

이걸 코드로 구현해야하는데.. 일단 막막하다..;;
뭔가 '해야겠다'라고 생각이 든것은

1. 반복되는 작업이 있다는게 눈에 띔
 - 반복문을 돌리거나 class를 만들자. 근데 어디까지 반복을 돌릴 수 있을까?
 - 반복되는 작업: output layer에서 w6, w7, w8 / hidden layer에서 w2, w3, w4
   output layer / hidden layer끼리 겹치는 작업이 있을까?

2. 수학적인 것들은 라이브러리가 있지 않을까?
 - 미분: sympy
 - 시그모이드: scipy > expit(), numpy

'''

# 라이브러리 임포트
from scipy import special as sp

# Forward

# 초기값 할당
i1 = 0.05
i2 = 0.1
w1 = 0.15
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.4
w6 = 0.45
w7 = 0.5
w8 = 0.55
o1 = 0.01
o2 = 0.99
b1 = 0.35
b2 = 0.6

# 함수(가중치1,노드1,가중치2,노드2,bias)

# 은닉계층 입력값 계산
net_h1 = w1*i1+w2*i2+b1*1
net_h2 = w3*i1+w4*i2+b1*1

# 활성화 함수
out_h1 = sp.expit(net_h1)
out_h2 = sp.expit(net_h2)

# 출력계층 입력값 계산
net_o1 = w5*out_h1+w6*out_h2*b2
net_o2 = w7*out_h1+w8*out_h2*b2

# 활성화 함수
out_o1 = sp.expit(net_o1)
out_o2 = sp.expit(net_o2)
