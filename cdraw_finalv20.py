﻿import cv2 as v,numpy as n,struct
z=n.ones((320,386,3),'u1')
z[:]=146,103,77
z[231:]=58,51,46
s='��*�"�2��o��o#}|�:"��C��)\x8e����}���b�\x8e�p炲�w�wBH�K7�:��1\0�2�\n�ju��Se|Ռ�qY�L~�M	dR�x�G�����i��D����M�U�0������Y��l	�z��V�@-�7�J�dJT��\x8ea:U���<m^����|}	M�����\n�x���`>~nD�	l�>�\0T}f@�\0T��G��	�b��3�ľb�Q�\n�p�J�"�>~H�K���\0T�%�T�d�-�V�ԈTz�V|}]}p\x8e�>�����p���@��S�G�����\n��T^�\0oJs�^�\'���	B|�|��Uct�U�\n�fA��g��w�\0�G�U�	m���	Tֺ|t��	���p����\x8eZ��ce�� �Xn�\rfi����ͮ�����\x8e>�t��!����>�5�4�J��k�\r_��|���}?Q[\x8e��[s���T�|���?���{	{��v��fM�6ͼ���;��p��\rI�����\nK�~��Ԩ����z�~:�v	�R�������|\x8e���*���b�O�G�	�T��ނ�*g� �?�}}bt�Q�����=uU�2�V�\'�\nh�`�!���XҤ���r��V'
m,h=512,62
for a,b,c,d,r in[struct.unpack('HHBBB',s[7*i:7*i+7])for i in range(90)]:
 x,y=a/m-h,b/m-h
 for k in range(h):v.circle(z,(a%m-h+x*k/h,b%m-h+y*k/h),r,n.clip((.9*c+.9*d-120,c-.3*d+41,.9*c-.6*d+80),0,255),-1)
v.imwrite('a.png',v.blur(z,(9,9)))