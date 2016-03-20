import cv2 as v,numpy as n,struct
z=n.ones((320,386,3),'u1')
z[:]=145,106,81
z[226:]=80,67,56
s='€v\n#„{~X©s"¾|ën²„ºÉ¬ª‹·.v"„4Á=ğŠ.•yv>ä;¦>t?ŞC°’GòS¾ó[pQ¹,g]x‰…gQÒWµlóš”´:eX K² }w¯hVƒZ[ÂbD¹t›¦\n§1˜‘±"}¼e®`h¸B½qòœ¥èJÕN©‘²f“­J¦üš	³|©t“|	\rÕ5SO©”¾zP±¤O€d…\rÆ¥L‹©|¸B{I¯Öb~¯½ÒdzQ½}‡}D«sŒ\x8ewx‡K	^pMz‘2L5`mce|ÙvlRc‹nš‚Jqw3œŸ|ÿ†G›Z:s4\r‚–]r.•	İX,(Œ\n*÷–“Œ¹òW’‰@Àà´IºäœQ,ŠpfuQhØvTzˆDÂ\\Nn•bSbº	|!1o0ˆ»5,‹f…Sñ8¿-“VÇ4}œ•¡$‹y 	­S(…Y³ek‰.†MÌ™ˆ 	wdvŸB\n …r†³UÆ¨JŠÒ^’š<©èf#}<©lux6º}€“\0S—ÑP{\0TB€Ïx°ƒ–A~w0’0ÃU‰)\x8e\n½Iá†\0Tò˜KUVmWOTæ¢y‚Œ’nLrŠXŒ„YˆKº\np„kJWÀw‘€«g"Sh—4‰kIg"„|[…p“Şœ£Š‹£ì$OH\\³>°nu9|6Õ¼†”¡.A2…œšqrÀ\\ZızE{m‘wG]—+Y•HÃŒèrälT·¥DNN\0T'
m,h=512,62
for a,b,c,d,r in[struct.unpack('HHBBB',s[7*i:7*i+7])for i in range(92)]:
 x,y=a/m-h,b/m-h
 for k in range(h):v.circle(z,(a%m-h+x*k/h,b%m-h+y*k/h),r,n.clip((.9*c+.9*d-120,c-.3*d+41,.9*c-.6*d+80),0,255),-1)
v.imwrite('a.png',v.blur(z,(9,9)))