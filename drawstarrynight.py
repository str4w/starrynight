﻿import cv2 as v,numpy as n,struct
z=n.ones((320,386,3),'u1')
z[:]=145,106,81
z[226:]=80,67,56
s='�v\n#��{~X�s"�|�n���ɬ���.v"�4�=��.�yv>�;�>t?�C���G�S��[pQ�,g]x��gQ�W�l󐚔�:eX�K� }w�hV�Z[�bD�t��\n��1���"}�e�`h�B�q��J�N����f��J�����	�|�t�|	\r�5SO����zP��O�d�\r�ƥL���|�B{I��b~���dzQ�}�}D�s�\x8ewx�K	^pMz�2L5`mce|�v�lRc�n��Jqw3��|��G�Z:s4�\r��]r.�	�X,�(�\n*������W��@��I��Q�,�pfuQh�vTz�D�\\Nn�bS�b�	|!1o0��5,�f�S�8�-�V�4}����$�y�	�S(�Y�ek�.�M̙���	�wdv�B\n��r��Uƨ�J���^��<��f#}<�lux6��}��\0S��P{\0TB��x���A~w0�0�U�)\x8e\n�I�\0T�KUVmWOT�y����nLr�X��Y�K�\np�kJW��w���g"Sh�4�kIg"�|�[�p�ޜ�����$OH\\�>�nu9|6ռ���.A2���qr�\\Z�zE{m�wG�]�+Y�HÌ�r�lT��DNN\0T'
m,h=512,62
for a,b,c,d,r in[struct.unpack('HHBBB',s[7*i:7*i+7])for i in range(92)]:
 x,y=a/m-h,b/m-h
 for k in range(h):v.circle(z,(a%m-h+x*k/h,b%m-h+y*k/h),r,n.clip((.9*c+.9*d-120,c-.3*d+41,.9*c-.6*d+80),0,255),-1)
v.imwrite('a.png',v.blur(z,(9,9)))