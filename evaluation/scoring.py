"""Retain literal WER and separately normalize Spanish numeric formatting.

No names, lexical substitutions, accents or negations are corrected. Integer
values 0..9999 and ordinary HH:MM formatting are expanded consistently for
both references and hypotheses. The two clock descriptions 'nueve y media'
and '9:30' are represented as the same 'nueve treinta' word sequence.
"""
import re
import unicodedata

SMALL=['cero','uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez','once','doce','trece','catorce','quince','dieciséis','diecisiete','dieciocho','diecinueve','veinte','veintiuno','veintidós','veintitrés','veinticuatro','veinticinco','veintiséis','veintisiete','veintiocho','veintinueve']
TENS={30:'treinta',40:'cuarenta',50:'cincuenta',60:'sesenta',70:'setenta',80:'ochenta',90:'noventa'}
HUNDREDS={2:'doscientos',3:'trescientos',4:'cuatrocientos',5:'quinientos',6:'seiscientos',7:'setecientos',8:'ochocientos',9:'novecientos'}

def integer(n):
    if not 0<=n<=9999:
        return str(n)
    if n<30:
        return SMALL[n]
    if n<100:
        return TENS[n//10*10]+(' y '+SMALL[n%10] if n%10 else '')
    if n==100:
        return 'cien'
    if n<1000:
        return ('ciento' if n//100==1 else HUNDREDS[n//100])+(' '+integer(n%100) if n%100 else '')
    return ('mil' if n//1000==1 else integer(n//1000)+' mil')+(' '+integer(n%1000) if n%1000 else '')

def tokens(text,normalize_numbers=False):
    text=unicodedata.normalize('NFC',text).casefold()
    if normalize_numbers:
        def clock(match):
            h,m=int(match[1]),int(match[2])
            if h>23 or m>59:
                return match[0]
            return integer(h)+((' '+integer(m)) if m else '')
        text=re.sub(r'\b(\d{1,2}):([0-5]\d)\b',clock,text)
        text=re.sub(r'\b\d+\b',lambda m:integer(int(m[0])),text)
        hours='|'.join(SMALL[1:13])
        text=re.sub(r'\b('+hours+r') y media\b',lambda m:m[1]+' treinta',text)
    return re.findall(r'\w+',text)

def errors(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def self_check():
    assert tokens('235 pesos',True)==tokens('doscientos treinta y cinco pesos',True)
    assert tokens('$15 pesos',True)==tokens('quince pesos',True)
    assert tokens('23 de noviembre',True)==tokens('veintitrés de noviembre',True)
    assert tokens('a las 9:30',True)==tokens('a las nueve y media',True)
    assert tokens('235 pesos',True)!=tokens('250 pesos',True)
    assert tokens('año',True)!=tokens('ano',True)
    assert tokens('No cierres la ventana',True)!=tokens('Cierra la ventana',True)
    assert errors(tokens('Si pagas en efectivo',True),tokens('Si paras en efectivo',True))==1

self_check()
