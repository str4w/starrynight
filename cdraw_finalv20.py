import cv2 as v,numpy as n,struct
z=n.ones((320,386,3),'u1')
z[:]=146,103,77
z[231:]=58,51,46
s='ƒ¼*¿"„2š…oš­o#}|ñ:"„¿Cğ€)\x8e€€™¼}¹Úb«\x8eÀpç‚²„w¢wBH·K7“:¦1\0•2š\n‘ju€‰Se|ÕŒœqYˆL~’M	dRƒx±G“†„šiµõD«‡™¦M´Uº0İù€œ”›Y‚‹l	¡z²’Vü@-7¢JdJT´ø\x8ea:U¹—<m^š—£|}	M¦ıœš’\nµx¥ˆ£`>~nD°	lˆ>ˆ\0T}f@’\0T†˜G†Ø	Òb”’3œÄ¾b‚Q½\n¿pœJ³"›>~HªK¶…–\0T¿%•Tªd¹-VÔˆTzŠV|}]}p\x8eÖ>—“›şpƒ³@½–S¯G¸¶‚¥\n½‚T^‹\0oJs€^Á\'ƒ„	B|Î|©…UctU‘\n»fA¦§gî”‰w¨\0ŸG…U’	m„²š	TÖº|tŠ•	œˆ¹pºŸ€ô\x8eZ£›ce¤ áXn…\rfi¨’•©Í®½œŒ¤î\x8e>‚t¢Š!ö’ >¸5‰4J¦ä”k¸\r_¤|Š«¶}?Q[\x8e ³[s—£¥T˜|‹¤½?¾¤·{	{†‹v‰“fM˜6Í¼†–œ;·İpŠ§\rIŠ’†Œ\nK€~„ªÔ¨£„†¨z~:¨v	ÉRˆˆ¡“ç|\x8e¤èÀ*—¢—b•O§G›	TªŠŞ‚Ü*g‡ ?î}}bt”Q®”€¹äš=uU¡2«V‹\'‹\nh¸`µ!ƒ¶‡XÒ¤†¡ƒrˆ²V'
m,h=512,62
for a,b,c,d,r in[struct.unpack('HHBBB',s[7*i:7*i+7])for i in range(90)]:
 x,y=a/m-h,b/m-h
 for k in range(h):v.circle(z,(a%m-h+x*k/h,b%m-h+y*k/h),r,n.clip((.9*c+.9*d-120,c-.3*d+41,.9*c-.6*d+80),0,255),-1)
v.imwrite('a.png',v.blur(z,(9,9)))