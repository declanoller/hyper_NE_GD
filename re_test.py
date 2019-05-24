import re



t = 'w_4_0_3_21'


def repl(mo):
    return(mo.group(1) + '_{' + mo.group(2) + ',' + mo.group(3) + ',' + mo.group(4) + ',' + mo.group(5) + '}')


print(re.sub(r'(w)_(\d+)_(\d+)_(\d+)_(\d+)', r'\1_{\2,\3,\4,\5}', t))
#print(re.sub(r'[w_\d+_\d+_\d+_\d+]', repl, t, count=0))




#
