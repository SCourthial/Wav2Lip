import numpy as np
import cv2

def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
  GA = np.float32(A.copy())
  GB = np.float32(B.copy())
  GM = np.float32(m.copy())
  print('GA', GA.shape)
  print('GB', GB.shape)
  print('GM', GM.shape)
  gpA = [GA]
  gpB = [GB]
  gpM = [GM]
  for i in range(num_levels):
    GA = cv2.pyrDown(GA)
    GB = cv2.pyrDown(GB)
    GM = cv2.pyrDown(GM)
    print('GA', GA.shape)
    print('GB', GB.shape)
    print('GM', GM.shape)
    gpA.append(np.float32(GA))
    gpB.append(np.float32(GB))
    gpM.append(np.float32(GM))

  lpA  = [gpA[num_levels-1]]
  lpB  = [gpB[num_levels-1]]
  gpMr = [gpM[num_levels-1]]
  for i in range(num_levels-1,0,-1):
    LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
    LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
    lpA.append(LA)
    lpB.append(LB)
    gpMr.append(gpM[i-1])

  LS = []
  for la,lb,gm in zip(lpA,lpB,gpMr):
    gm = gm[:,:,np.newaxis]
    ls = la * gm + lb * (1.0 - gm)
    LS.append(ls)

  ls_ = LS[0]
  for i in range(1,num_levels):
    ls_ = cv2.pyrUp(ls_)
    print('ls_', ls_.shape, ls_.dtype)
    print('LS[i]', LS[i].shape, LS[i].dtype)
    ls_ = cv2.add(ls_, LS[i])
  return ls_